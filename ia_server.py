import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time, random, math, os, threading, csv, signal
import io, gzip, base64, inspect, uuid, heapq, re as _re, json as _json
from collections import deque
from rich.console import Console
try:
    import requests
except ImportError:
    requests = None
try:
    from flask import Flask, request, jsonify, send_file
except ImportError:
    Flask = None
console = Console()
_server_lock = threading.RLock()  # RLock: precisa ser reentrante (algumas funções federadas se chamam em cadeia já segurando o lock)
SEED = 42
CONTEXT_LEN = 1024 # era 64 → 128 → 512 — contexto maior treinado via Colab, bs=1 compensado pela agregação federada
PRETRAIN_STEPS = 50000  # teto de segurança — o treino normalmente para antes, por loss (abaixo)
# --- parada por loss (hardcoded) — calibrado pra ~100-200MB de texto, modelo ~15M params, BPE 4096 ---
# Loss de um batch (bs=4) é ruidosa, então tudo é medido na média móvel das últimas LOSS_WINDOW steps.
LOSS_WINDOW = 200            # janela da média móvel
LOSS_MIN_STEPS = 1000        # nunca para antes disso (evita parar num vale falso no começo)
LOSS_TARGET = 3.3            # para se a média móvel chegar aqui (estimativa do piso pra esse tamanho de modelo/vocab)
LOSS_PLATEAU_PATIENCE = 2000 # para se a média móvel não melhorar por essa quantidade de steps...
LOSS_PLATEAU_MIN_DELTA = 0.01  # ...onde "melhorar" = cair pelo menos isso em relação ao melhor já visto
PRETRAIN_LR = 0.0006  # era 0.003 — alto demais pra ~14M params, arriscava instabilidade
MOE_AUX_LOSS_COEF = 0.01  # peso da load-balancing loss (Switch Transformer) — evita
                          # colapso de roteamento (poucos experts recebendo quase
                          # todo o tráfego de tokens) sem competir com a loss principal
PRETRAIN_LR_CONT = 0.0003  # era 0.0008
REINFORCE_LR = 0.0005
SUPERVISED_LR = 0.0005  # era 0.001
# --- treino local por página (--training-local) ---
# Antes: loss_target=1.8, max_steps_por_pagina=2000 — perseguir um loss tão
# baixo numa única página, reamostrando só ela por até 2000 passos, decora o
# texto em vez de generalizar (catastrophic forgetting das páginas
# anteriores). Meta mais frouxa + menos passos + replay buffer abaixo
# resolvem isso.
LOCAL_TRAIN_LOSS_TARGET = 2.5       # era 1.8
LOCAL_TRAIN_MAX_STEPS_POR_PAGINA = 300  # era 2000
REPLAY_BUFFER_MAX_PAGES = 30         # quantas páginas recentes ficam disponíveis pra replay
REPLAY_BATCH_SIZE = 4                # tamanho do batch por step de treino local (era 1)
REPLAY_FRACTION = 0.5                # fração do batch vinda de páginas antigas (resto = página atual)
TEMP_START = 1.1
_KAGGLE_WORKING = '/kaggle/working'
if os.path.isdir(_KAGGLE_WORKING):
    CHECKPOINT_DIR = os.path.join(_KAGGLE_WORKING, 'checkpoints')
else:
    CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_EVERY = 50
CHECKPOINT_LAST = 'checkpoint_last.pt'
INIT_CONFIG = dict(embed_dim=256, n_heads=8, n_layers=6, dropout=0.1, use_moe=True, n_experts=4, top_k=1)
# Tier 1 (~14M params totais, ~poucos M ativos por token já que top_k=1).
# Tier 2 (~43M): embed_dim=384, n_heads=8,  n_layers=8,  n_experts=4, top_k=1, CONTEXT_LEN=320
# Tier 3 (~95M): embed_dim=512, n_heads=8,  n_layers=10, n_experts=4, top_k=1, CONTEXT_LEN=384
EXPAND_EMBED_DELTA = 16
torch.manual_seed(SEED)
random.seed(SEED)
CSV_MAX_CHARS_POR_ARQUIVO = 2000000
CSV_SAMPLE_LINHAS = 200
CSV_MIN_LEN_MEDIO_TEXTO = 15

# LIMITE MÁXIMO DE ARQUIVOS E CARACTERES (NOVO - RAILWAY FIX)
CORPUS_MAX_FILES = 50  # Máximo 50 arquivos
CORPUS_MAX_CHARS_TOTAL = 500000000  # 5MB total
IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.env', 'dist', 'build', '.pytest_cache', '.tox', '.idea', '.vscode'}

def _extrair_texto_csv(caminho_completo):
    texto = []
    total_chars = 0
    try:
        with open(caminho_completo, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            amostra = [f.readline() for _ in range(CSV_SAMPLE_LINHAS)]
            f.seek(0)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(''.join(amostra[:20]) or ',')
            except Exception:
                dialect = csv.excel
            reader = csv.reader(f, dialect)
            header = next(reader, None)
            if header is None:
                return ''
            linhas_amostra = []
            for i, row in enumerate(reader):
                linhas_amostra.append(row)
                if i >= CSV_SAMPLE_LINHAS:
                    break
            n_cols = len(header)
            colunas_texto = set()
            for c in range(n_cols):
                vals = [row[c] for row in linhas_amostra if c < len(row)]
                if not vals:
                    continue
                media = sum((len(v) for v in vals)) / len(vals)
                if media >= CSV_MIN_LEN_MEDIO_TEXTO:
                    colunas_texto.add(c)
            if not colunas_texto:
                colunas_texto = set(range(n_cols))
            f.seek(0)
            reader = csv.reader(f, dialect)
            next(reader, None)
            for row in reader:
                partes = [row[c] for c in sorted(colunas_texto) if c < len(row)]
                linha_txt = ' '.join((p for p in partes if p))
                if not linha_txt:
                    continue
                texto.append(linha_txt)
                total_chars += len(linha_txt) + 1
                if total_chars >= CSV_MAX_CHARS_POR_ARQUIVO:
                    break
    except Exception as e:
        print(f'Erro ao ler CSV {os.path.basename(caminho_completo)}: {e}')
        return ''
    return '\n'.join(texto)

def carregar_corpus_categorizado(diretorios_raiz, categorias: dict):
    if isinstance(diretorios_raiz, str):
        diretorios_raiz = [diretorios_raiz]
    resultado = {nome: '' for nome in categorias}
    ext_para_categoria = {}
    for nome, exts in categorias.items():
        for ext in exts:
            ext_para_categoria[ext] = nome
    
    arquivo_count = 0  # CONTADOR (NOVO)
    total_chars = 0    # TOTAL CHARS (NOVO)
    
    for diretorio_raiz in diretorios_raiz:
        if not os.path.isdir(diretorio_raiz):
            continue
        for pasta_atual, subpastas, arquivos in os.walk(diretorio_raiz):
            # IGNORA DIRETÓRIOS PERIGOSOS (NOVO)
            subpastas[:] = [d for d in subpastas if d not in IGNORE_DIRS]
            for nome_arquivo in arquivos:
                # PAROU SE ATINGIU LIMITE (NOVO)
                if arquivo_count >= CORPUS_MAX_FILES or total_chars >= CORPUS_MAX_CHARS_TOTAL:
                    print(f'⚠️  Limite de corpus atingido ({arquivo_count} arquivos, {total_chars} chars)')
                    return resultado
                
                ext_match = next((e for e in ext_para_categoria if nome_arquivo.endswith(e)), None)
                if ext_match is None:
                    continue
                if nome_arquivo in ('self_evolving_ai-1.py', 'self_evolving_ai_v2.py'):
                    continue
                caminho_completo = os.path.join(pasta_atual, nome_arquivo)
                categoria = ext_para_categoria[ext_match]
                try:
                    if nome_arquivo.lower().endswith('.csv'):
                        trecho = _extrair_texto_csv(caminho_completo)
                        if trecho:
                            resultado[categoria] += trecho + '\n'
                            total_chars += len(trecho)
                            arquivo_count += 1
                            print(f'Lido (CSV, {categoria}): {nome_arquivo} ({len(trecho)} chars)')
                        else:
                            print(f'CSV vazio/ignorado: {nome_arquivo}')
                    else:
                        with open(caminho_completo, 'r', encoding='utf-8', errors='ignore') as f:
                            conteudo = f.read()
                            resultado[categoria] += conteudo + '\n'
                            total_chars += len(conteudo)
                            arquivo_count += 1
                            print(f'Lido ({categoria}): {nome_arquivo} ({len(conteudo)} chars)')
                except Exception as e:
                    print(f'Erro ao ler {nome_arquivo}: {e}')
    return resultado
try:
    diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
except NameError:
    diretorio_raiz = os.getcwd()
# IGNORA /kaggle/input se não existir (NOVO)
EXTRA_CORPUS_DIRS = ['/kaggle/input'] if os.path.isdir('/kaggle/input') else []
_CATEGORIAS_CORPUS = {'general': ('.txt', '.csv')}
# 'code': ('.py',) foi removido por enquanto — estava fazendo o próprio
# app.py/ia_server.py (e qualquer outro .py da pasta) entrarem no corpus
# de treino junto com os datasets de texto, contaminando o aprendizado de
# inglês básico com sintaxe/identificadores Python (CONTEXT_LEN, state_dict,
# console.print, etc. apareciam nas gerações). Reative quando for a hora de
# focar em código de propósito, e idealmente com um corpus de código
# dedicado (como o fine_tuning_dataset.csv), não o código-fonte do projeto.
_corpus_por_categoria = carregar_corpus_categorizado([diretorio_raiz] + EXTRA_CORPUS_DIRS, _CATEGORIAS_CORPUS)
CORPUS_CODE = _corpus_por_categoria.get('code', '')  # vazio enquanto 'code' não estiver em _CATEGORIAS_CORPUS
CORPUS_GENERAL = _corpus_por_categoria['general']
CORPUS = CORPUS_CODE + '\n' + CORPUS_GENERAL
EOS = '\x00'
# ═══════════════════════════════════════════════════════════════════════
# TOKENIZER — BPE byte-level (substitui o char-level anterior)
# ═══════════════════════════════════════════════════════════════════════
# Por que byte-level: a base do vocab são os 256 valores de byte, então
# QUALQUER texto UTF-8 é representável sem nunca cair num "caractere não
# visto" — a mesma garantia que o char-level tinha (via i2c.get(i,'?') e o
# crescimento dinâmico de vocab), só que por construção, sem precisar
# redimensionar tok_emb/head em tempo real. Cada token passa a cobrir em
# média ~3-4 bytes de texto real, então CONTEXT_LEN (em tokens) passa a
# enxergar bem mais contexto do que via char puro — esse é o ganho que
# motivou a migração.
BPE_VOCAB_SIZE = int(os.environ.get('BPE_VOCAB_SIZE', 4096))
# 4096 é um meio-termo pro tier ~14M params (INIT_CONFIG: embed_dim=256):
# tok_emb + head somados ficam em ~4096*256*2 ≈ 2M params extras (~15% do
# modelo) — vocab maior comprime mais texto por posição, mas infla essas
# duas camadas. Se for controlar o tamanho total, mexa aqui (e no cache
# de tokenizer.json, que precisa ser apagado pra retreinar com vocab novo).

def _bpe_get_pair_counts(ids, alive, nxt):
    counts = {}
    pos = {}
    for i in range(len(ids)):
        if not alive[i]:
            continue
        j = nxt[i]
        if j == -1:
            continue
        p = (ids[i], ids[j])
        counts[p] = counts.get(p, 0) + 1
        pos.setdefault(p, set()).add(i)
    return counts, pos

def _bpe_train(text: str, vocab_size: int) -> dict:
    """Treina merges de BPE byte-level. Usa lista encadeada (prev/next) +
    heap de contagens em vez de reescanear o corpus inteiro a cada merge
    (isso seria O(n * n_merges) — inviável pra corpus de alguns MB; aqui
    cada merge custa só o tamanho da vizinhança afetada)."""
    raw = text.encode('utf-8')
    n = len(raw)
    if n < 2:
        return {}
    ids = list(raw)
    nxt = list(range(1, n)) + [-1]
    prv = list(range(-1, n - 1))
    alive = [True] * n
    pair_count, pair_pos = _bpe_get_pair_counts(ids, alive, nxt)
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)
    merges = {}
    next_id = 256
    EOS_BYTE = 0  # nunca funde o byte 0 (EOS) com vizinhos — mantém o EOS
                  # sempre como token atômico, igual garantia que o char-level
                  # dava reservando um índice próprio pra ele no vocab
    while next_id < vocab_size:
        chosen = None
        while heap:
            negc, p = heapq.heappop(heap)
            if pair_count.get(p, 0) == -negc and -negc >= 2:
                chosen = p
                break
        if chosen is None:
            break
        if EOS_BYTE in chosen:
            pair_count[chosen] = 0
            continue
        positions = list(pair_pos.get(chosen, ()))
        merges[chosen] = next_id
        for i in positions:
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or ids[i] != chosen[0] or ids[j] != chosen[1]:
                continue
            p_, n_ = (prv[i], nxt[j])
            if p_ != -1:
                old = (ids[p_], ids[i])
                pair_count[old] = pair_count.get(old, 0) - 1
                pair_pos.get(old, set()).discard(p_)
            if n_ != -1:
                old = (ids[j], ids[n_])
                pair_count[old] = pair_count.get(old, 0) - 1
                pair_pos.get(old, set()).discard(j)
            ids[i] = next_id
            alive[j] = False
            nxt[i] = n_
            if n_ != -1:
                prv[n_] = i
            if p_ != -1:
                newp = (ids[p_], ids[i])
                pair_count[newp] = pair_count.get(newp, 0) + 1
                pair_pos.setdefault(newp, set()).add(p_)
                heapq.heappush(heap, (-pair_count[newp], newp))
            if n_ != -1:
                newp = (ids[i], ids[n_])
                pair_count[newp] = pair_count.get(newp, 0) + 1
                pair_pos.setdefault(newp, set()).add(i)
                heapq.heappush(heap, (-pair_count[newp], newp))
        pair_count[chosen] = 0
        next_id += 1
    return merges

def _bpe_vocab_from_merges(merges: dict) -> dict:
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in sorted(merges.items(), key=lambda kv: kv[1]):
        vocab[idx] = vocab[a] + vocab[b]
    return vocab

def _bpe_encode(text: str, merges: dict) -> list:
    """Mesma ideia do _bpe_train: lista encadeada (prev/next) + heap de
    prioridade, em vez de reconstruir a lista inteira a cada merge aplicado.
    A versão antiga fazia set(zip(ids, ids[1:])) + reescrever a lista toda
    por merge — O(n_merges * n), inviável pra texto grande (o CORPUS
    inteiro, ~200KB, com ~3840 merges). Aqui cada merge custa só a
    vizinhança afetada, igual no treino: O(n + n_merges_aplicados).
    Resultado idêntico ao algoritmo original (sempre aplica o merge de
    menor id/maior prioridade disponível a cada rodada)."""
    ids = list(text.encode('utf-8'))
    n = len(ids)
    if n < 2 or not merges:
        return ids
    nxt = list(range(1, n)) + [-1]
    prv = list(range(-1, n - 1))
    alive = [True] * n
    heap = []
    for i in range(n - 1):
        pair = (ids[i], ids[i + 1])
        rank = merges.get(pair)
        if rank is not None:
            heapq.heappush(heap, (rank, i))
    while heap:
        rank, i = heapq.heappop(heap)
        if not alive[i]:
            continue
        j = nxt[i]
        if j == -1:
            continue
        pair = (ids[i], ids[j])
        if merges.get(pair) != rank:
            continue  # posição desatualizada (par mudou desde que foi empilhado)
        p_, n_ = (prv[i], nxt[j])
        ids[i] = rank
        alive[j] = False
        nxt[i] = n_
        if n_ != -1:
            prv[n_] = i
        # novo par à esquerda (pode ter surgido/mudado prioridade)
        if p_ != -1:
            newp = (ids[p_], ids[i])
            newrank = merges.get(newp)
            if newrank is not None:
                heapq.heappush(heap, (newrank, p_))
        # novo par à direita
        if n_ != -1:
            newp = (ids[i], ids[n_])
            newrank = merges.get(newp)
            if newrank is not None:
                heapq.heappush(heap, (newrank, i))
    out = []
    i = 0
    while i != -1:
        if alive[i]:
            out.append(ids[i])
        i = nxt[i]
    return out

def _bpe_decode(ids: list, vocab: dict) -> str:
    b = b''.join((vocab.get(i, b'?') for i in ids))
    return b.decode('utf-8', errors='replace')

def _bpe_merges_to_json(merges: dict) -> list:
    return [[list(p), i] for p, i in merges.items()]

def _bpe_merges_from_json(data: list) -> dict:
    return {tuple(p): i for p, i in data}

def _bpe_load_or_train(corpus_text: str, vocab_size: int, cache_path: str):
    """Tenta carregar merges já treinados do disco antes de retreinar —
    treinar BPE num corpus de alguns MB não é instantâneo, então cachear
    evita pagar esse custo a cada restart do processo."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                saved = _json.load(f)
            if saved.get('vocab_size') == vocab_size:
                merges = _bpe_merges_from_json(saved['merges'])
                return merges, _bpe_vocab_from_merges(merges)
        except Exception:
            pass
    console.print(f'[bold cyan]🔤 Treinando tokenizer BPE (vocab_size={vocab_size})...[/bold cyan]')
    merges = _bpe_train(corpus_text, vocab_size)
    vocab = _bpe_vocab_from_merges(merges)
    try:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            _json.dump({'vocab_size': vocab_size, 'merges': _bpe_merges_to_json(merges)}, f)
    except Exception as e:
        console.print(f'[yellow]⚠️  Não deu pra salvar cache do tokenizer: {e}[/yellow]')
    console.print(f'[bold cyan]🔤 Tokenizer BPE pronto: {len(vocab)} tokens ({len(merges)} merges)[/bold cyan]')
    return merges, vocab

_TOKENIZER_CACHE_PATH = os.path.join(CHECKPOINT_DIR, 'tokenizer.json')
BPE_MERGES, BPE_VOCAB = _bpe_load_or_train(CORPUS, BPE_VOCAB_SIZE, _TOKENIZER_CACHE_PATH)
VOCAB = len(BPE_VOCAB)
encode = lambda s: _bpe_encode(s, BPE_MERGES)
decode = lambda ids: _bpe_decode(ids, BPE_VOCAB)
NEWLINE_ID = encode('\n')[0]  # 1 byte só → nunca passa por merge, sempre atômico
EOS_ID = 0                    # byte 0, protegido de merges em _bpe_train
data_tensor = torch.tensor(encode(CORPUS), dtype=torch.long)
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1000000000.0
    dev_str = f'🚀 {gpu_name} ({vram_gb:.1f}GB VRAM)'
else:
    device = torch.device('cpu')
    dev_str = '⚠️  CPU (sem GPU detectada)'
    try:
        _n_cpu = os.cpu_count() or 1
        torch.set_num_threads(_n_cpu)
        torch.set_num_interop_threads(max(1, _n_cpu // 2))
    except Exception:
        pass

class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim, n_heads, dropout, ctx):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.hd = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ad = nn.Dropout(dropout)
        self.rd = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(ctx, ctx)).view(1, 1, ctx, ctx)
        self.register_buffer('mask', mask)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        split = lambda t: t.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        q, k, v = (split(q), split(k), split(v))
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v) if use_cache else None
        Tk = k.shape[2]
        w = q @ k.transpose(-2, -1) / math.sqrt(self.hd)
        if past_kv is None:
            w = w.masked_fill(self.mask[:, :, :T, :Tk] == 0, float('-inf'))
        w = self.ad(F.softmax(w, dim=-1))
        o = (w @ v).transpose(1, 2).contiguous().view(B, T, C)
        return (self.rd(self.out(o)), new_kv)

class MoELayer(nn.Module):

    def __init__(self, embed_dim: int, n_experts: int=4, top_k: int=2, dropout: float=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.router = nn.Linear(embed_dim, n_experts, bias=False)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout)) for _ in range(n_experts)])
        # Load-balancing (estilo Switch Transformer): guarda a loss auxiliar do
        # último forward pra ser somada na loss principal depois. Sem isso, o
        # roteador só aprende pelo gradiente da tarefa e pode colapsar num
        # sub-conjunto de experts, desperdiçando a capacidade dos outros.
        self.last_aux_loss = None

    def forward(self, x: torch.Tensor, forced_expert: int=None) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)
        if forced_expert is not None:
            expert_out = self.experts[forced_expert](x_flat)
            return expert_out.reshape(B, T, C)
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = router_probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-09)

        # --- load-balancing loss (Switch Transformer, eq. 4-5) ---
        # f_i = fração de tokens roteados pro expert i (contagem "dura", via top-1)
        # P_i = probabilidade média que o roteador deu ao expert i (soft, com grad)
        # aux_loss = n_experts * sum(f_i * P_i) -> mínimo quando distribuição é uniforme
        n_tok = x_flat.shape[0]
        top1_idx = topk_idx[:, 0]
        f_i = torch.zeros(self.n_experts, device=x.device, dtype=router_probs.dtype)
        f_i.scatter_add_(0, top1_idx, torch.ones(n_tok, device=x.device, dtype=router_probs.dtype))
        f_i = f_i / max(n_tok, 1)
        P_i = router_probs.mean(dim=0)
        self.last_aux_loss = self.n_experts * (f_i * P_i).sum()

        output = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            is_selected = topk_idx == expert_id
            token_mask = is_selected.any(dim=-1)
            if not token_mask.any():
                continue
            weight = is_selected[token_mask].float() * topk_probs[token_mask]
            weight = weight.sum(dim=-1, keepdim=True)
            expert_out = expert(x_flat[token_mask])
            output[token_mask] += expert_out * weight
        return output.reshape(B, T, C)

    def expert_load(self) -> torch.Tensor:
        return torch.ones(self.n_experts) / self.n_experts

class Block(nn.Module):

    def __init__(self, embed_dim, n_heads, dropout, ctx, use_moe=False, n_experts=4, top_k=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout, ctx)
        self.ln2 = nn.LayerNorm(embed_dim)
        if use_moe:
            self.mlp = MoELayer(embed_dim, n_experts=n_experts, top_k=top_k, dropout=dropout)
        else:
            self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x, past_kv=None, use_cache=False, forced_expert=None):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        mlp_in = self.ln2(x)
        if forced_expert is not None and isinstance(self.mlp, MoELayer):
            x = x + self.mlp(mlp_in, forced_expert=forced_expert)
        else:
            x = x + self.mlp(mlp_in)
        return (x, new_kv)

class TinyAI(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        ed = cfg['embed_dim']
        nh = cfg['n_heads']
        nl = cfg['n_layers']
        do = cfg['dropout']
        use_moe = cfg.get('use_moe', False)
        ne = cfg.get('n_experts', 4)
        tk = cfg.get('top_k', 2)
        self.tok_emb = nn.Embedding(VOCAB, ed)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, ed)
        self.drop = nn.Dropout(do)
        self.blocks = nn.ModuleList([Block(ed, nh, do, CONTEXT_LEN, use_moe=use_moe, n_experts=ne, top_k=tk) for _ in range(nl)])
        self.ln_f = nn.LayerNorm(ed)
        self.head = nn.Linear(ed, VOCAB, bias=False)
        self.apply(self._init)
        self._cfg = cfg

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, past_kv=None, use_cache=False, forced_expert=None):
        B, T = idx.shape
        past_len = past_kv[0][0].shape[2] if past_kv is not None else 0
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pkv = past_kv[i] if past_kv is not None else None
            x, nkv = block(x, past_kv=pkv, use_cache=use_cache, forced_expert=forced_expert)
            if use_cache:
                new_kvs.append(nkv)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
            # soma a loss de load-balancing de cada camada MoE (se houver).
            # Coeficiente pequeno (0.01, igual ao paper do Switch Transformer)
            # pra não competir com a loss principal de predição de token.
            aux_losses = [b.mlp.last_aux_loss for b in self.blocks
                          if isinstance(b.mlp, MoELayer) and b.mlp.last_aux_loss is not None]
            if aux_losses:
                loss = loss + MOE_AUX_LOSS_COEF * torch.stack(aux_losses).mean()
        return (logits, loss, new_kvs)

    @property
    def n_params(self):
        return sum((p.numel() for p in self.parameters()))

# ═══════════════════════════════════════════════════════════════════════
# TREINO DISTRIBUÍDO (voluntários) — protocolo FedAvg simplificado
# ═══════════════════════════════════════════════════════════════════════
# Fluxo:
#   1. Voluntário pede GET /v1/job          -> recebe pesos atuais (comprimidos),
#      config do modelo, código-fonte das classes do modelo, um pedaço de
#      texto (ou nada, se ele for treinar com conteúdo próprio) e quantos
#      steps rodar.
#   2. Voluntário treina localmente N steps e calcula delta = pesos_depois - pesos_antes.
#   3. Voluntário manda POST /v1/submit com o delta comprimido.
#   4. Servidor acumula deltas de vários voluntários; ao atingir o limiar
#      (ou o timeout), faz a média ponderada por n_steps e aplica no modelo
#      mestre, incrementando a "version". Deltas de uma "version" antiga
#      (o mestre já mudou desde que o job foi emitido) são descartados —
#      staleness simples, evita puxar o treino pra trás.
FED_JOB_CONTENT_CHARS = 20000       # tamanho do pedaço de corpus mandado por job
FED_MAX_STEPS_PER_JOB = 150         # teto de steps que um voluntário pode rodar por job
FED_AGG_THRESHOLD = 1               # aplica o delta assim que chega — evita acumular vários em RAM
FED_JOB_TTL_SECONDS = 1800          # jobs emitidos e nunca respondidos expiram
FED_MAX_PENDING_JOBS = 500          # limite de jobs "em aberto" guardados em memória

_fed_lock = threading.Lock()
_VOCAB_EPOCH = 0  # incrementado toda vez que o vocab muda — sinaliza pra quem tem
                  # um optimizer vivo que ele precisa ser reconstruído (senão fica
                  # segurando referência aos tensores antigos de tok_emb/head pra
                  # sempre, vazando memória a cada expansão)
_fed_state = {
    'version': 0,
    'jobs': {},            # job_id -> {'version', 'issued_at', 'worker_id_hint'}
    'pending_deltas': [],  # lista de (delta_state_dict, n_steps, worker_id)
    'stats': {'jobs_issued': 0, 'jobs_submitted': 0, 'jobs_rejected_stale': 0, 'aggregations': 0},
}


def _compress_state_dict(sd: dict) -> str:
    buf = io.BytesIO()
    torch.save(sd, buf)
    comp = gzip.compress(buf.getvalue(), compresslevel=6)
    return base64.b64encode(comp).decode('ascii')


def _decompress_state_dict(b64_str: str, weights_only: bool = True) -> dict:
    comp = base64.b64decode(b64_str)
    raw = gzip.decompress(comp)
    buf = io.BytesIO(raw)
    # weights_only=True restringe o unpickling a tensores/tipos seguros —
    # importante aqui porque esse payload vem de gente estranha na internet.
    return torch.load(buf, map_location='cpu', weights_only=weights_only)


_MODEL_SOURCE_CACHE = None


def _get_model_source() -> str:
    """Código-fonte das classes do modelo, mandado pro voluntário executar
    localmente (exec) pra reconstruir a arquitetura sem precisar do arquivo
    inteiro de 1300+ linhas nem manter um segundo arquivo sincronizado."""
    global _MODEL_SOURCE_CACHE
    if _MODEL_SOURCE_CACHE is None:
        parts = [inspect.getsource(cls) for cls in (CausalSelfAttention, MoELayer, Block, TinyAI)]
        _MODEL_SOURCE_CACHE = '\n\n'.join(parts)
    return _MODEL_SOURCE_CACHE


def _fed_cleanup_expired_jobs():
    now = time.time()
    expired = [jid for jid, j in _fed_state['jobs'].items() if now - j['issued_at'] > FED_JOB_TTL_SECONDS]
    for jid in expired:
        del _fed_state['jobs'][jid]
    if len(_fed_state['jobs']) > FED_MAX_PENDING_JOBS:
        # descarta os mais velhos se acumular lixo demais (voluntários que sumiram)
        oldest = sorted(_fed_state['jobs'].items(), key=lambda kv: kv[1]['issued_at'])
        for jid, _ in oldest[:len(_fed_state['jobs']) - FED_MAX_PENDING_JOBS]:
            del _fed_state['jobs'][jid]


def _fed_issue_job(model: 'TinyAI', own_content: bool) -> dict:
    with _fed_lock:
        _fed_cleanup_expired_jobs()
        job_id = uuid.uuid4().hex
        version = _fed_state['version']
        _fed_state['jobs'][job_id] = {'version': version, 'issued_at': time.time()}
        _fed_state['stats']['jobs_issued'] += 1
        weights_b64 = _compress_state_dict(model.state_dict())
    payload = {
        'job_id': job_id,
        'version': version,
        'model_cfg': model._cfg,
        'model_code': _get_model_source(),
        'bpe_merges': _bpe_merges_to_json(BPE_MERGES),
        'context_len': CONTEXT_LEN,
        'weights': weights_b64,
        'max_steps': FED_MAX_STEPS_PER_JOB,
    }
    if not own_content:
        if _WIKI_MODE:
            try:
                title, texto = _wiki_take_page()
                payload['content'] = texto[:FED_JOB_CONTENT_CHARS]
                payload['content_source'] = f'wikipedia:{title}'
            except Exception as e:
                console.print(f'[bold yellow]⚠️  Wikipedia falhou nesse job ({e}), caindo pro corpus local[/bold yellow]')
                own_content = False  # cai no bloco abaixo via fallback manual
        if not _WIKI_MODE or 'content' not in payload:
            if len(CORPUS) > FED_JOB_CONTENT_CHARS:
                start = random.randint(0, len(CORPUS) - FED_JOB_CONTENT_CHARS)
                payload['content'] = CORPUS[start:start + FED_JOB_CONTENT_CHARS]
            else:
                payload['content'] = CORPUS
            payload.setdefault('content_source', 'local_corpus')
    return payload


def _fed_submit_delta(model: 'TinyAI', job_id: str, version: int, n_steps: int, delta_b64: str, worker_id: str) -> dict:
    with _fed_lock:
        job = _fed_state['jobs'].pop(job_id, None)
        if job is None:
            return {'accepted': False, 'reason': 'job_id desconhecido ou expirado'}
        if version != _fed_state['version']:
            _fed_state['stats']['jobs_rejected_stale'] += 1
            return {'accepted': False, 'reason': f"versão desatualizada (mestre já é v{_fed_state['version']})", 'current_version': _fed_state['version']}
        if n_steps <= 0 or n_steps > FED_MAX_STEPS_PER_JOB:
            return {'accepted': False, 'reason': 'n_steps inválido'}
    try:
        delta = _decompress_state_dict(delta_b64, weights_only=True)
    except Exception as e:
        return {'accepted': False, 'reason': f'delta corrompido/ilegível: {e}'}
    ref_sd = model.state_dict()
    for k, v in delta.items():
        if k not in ref_sd or v.shape != ref_sd[k].shape:
            return {'accepted': False, 'reason': f'shape incompatível em {k} — modelo do voluntário está com config diferente do mestre'}
    should_aggregate = False
    with _fed_lock:
        _fed_state['pending_deltas'].append((delta, n_steps, worker_id))
        _fed_state['stats']['jobs_submitted'] += 1
        if len(_fed_state['pending_deltas']) >= FED_AGG_THRESHOLD:
            should_aggregate = True
    new_version = _fed_state['version']
    if should_aggregate:
        new_version = _fed_apply_pending(model)
    return {'accepted': True, 'aggregated': should_aggregate, 'version': new_version}


def _fed_apply_pending(model: 'TinyAI') -> int:
    """Aplica os deltas pendentes direto nos tensores do modelo (in-place,
    sob torch.no_grad()) em vez de montar cópias extras (zeros_like por
    parâmetro + dict novo pro load_state_dict) — importante em container
    com pouca RAM, onde cada cópia extra de ~50MB+ pode ser a diferença
    entre rodar e OOM."""
    with _fed_lock:
        deltas = _fed_state['pending_deltas']
        _fed_state['pending_deltas'] = []
        if not deltas:
            return _fed_state['version']
        total_steps = sum(n for _, n, _ in deltas)
        with _server_lock, torch.no_grad():
            sd = model.state_dict()  # tensores compartilhados com o modelo, não cópias
            for key, tensor in sd.items():
                if not torch.is_floating_point(tensor):
                    continue  # não mistura buffers inteiros (ex: máscara causal) na média
                for delta, n_steps, _worker in deltas:
                    if key in delta:
                        tensor.add_(delta[key].to(tensor.dtype), alpha=n_steps / total_steps)
        del sd
        _fed_state['version'] += 1
        _fed_state['stats']['aggregations'] += 1
        contributors = [w for _, _, w in deltas]
        console.print(f"[bold magenta]🤝 Agregados {len(deltas)} deltas ({total_steps} steps totais) de {contributors} → modelo agora é v{_fed_state['version']}[/bold magenta]")
        return _fed_state['version']


def _ckpt_path(filename: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, filename)

def save_checkpoint(model: TinyAI, rl_opt, sup_opt, state: dict, filename: str=CHECKPOINT_LAST, pretrain_opt=None) -> None:
    path = _ckpt_path(filename)
    payload = {'model_cfg': model._cfg, 'model_state': model.state_dict(), 'bpe_merges': _bpe_merges_to_json(BPE_MERGES), 'vocab_size': VOCAB, 'context_len': CONTEXT_LEN, 'rl_opt_state': rl_opt.state_dict(), 'sup_opt_state': sup_opt.state_dict(), 'pretrain_opt_state': pretrain_opt.state_dict() if pretrain_opt is not None else None, 'gen': state['gen'], 'temp': state['temp'], 'best_reward': state['best_reward'], 'best_code': state['best_code'], 'best_gen': state['best_gen'], 'config': state['config'], 'mutation_log': state['mutation_log'], 'reward_hist': list(state['reward_hist']), 'temp_resets': state['temp_resets'], 'stdout_memory': list(state.get('stdout_memory', [])), 'stdout_norm_mem': list(state.get('stdout_norm_memory', []))}
    torch.save(payload, path)

def load_checkpoint(filename: str=CHECKPOINT_LAST):
    path = _ckpt_path(filename)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt

def _transplant_state_dict(saved_sd: dict, model: 'TinyAI', skip_prefixes: tuple = ()) -> None:
    dst_sd = model.state_dict()
    for key in dst_sd:
        if key not in saved_sd:
            continue
        if any(key.startswith(p) for p in skip_prefixes):
            continue  # ex: tok_emb./head. quando o esquema de vocab mudou —
                       # transplantar essas linhas seria misturar índices de
                       # tokenizers diferentes, pior que reiniciar do zero
        s, d = (saved_sd[key], dst_sd[key])
        if s.shape == d.shape:
            dst_sd[key] = s.clone()
        else:
            slices = tuple((slice(0, min(a, b)) for a, b in zip(s.shape, d.shape)))
            dst_sd[key][slices] = s[slices].clone()
    model.load_state_dict(dst_sd)

def restore_from_checkpoint(ckpt: dict, state: dict):
    global BPE_MERGES, BPE_VOCAB, VOCAB, data_tensor
    legacy_char_ckpt = 'bpe_merges' not in ckpt and 'vocab_chars' in ckpt
    if 'bpe_merges' in ckpt:
        # Reconstrói o tokenizer EXATO usado quando esse checkpoint foi
        # salvo, em vez de confiar no que foi treinado nessa sessão a partir
        # do corpus local. Isso corrige uma fragilidade que o char-level
        # tinha: o vocab char-level era recriado do zero a cada restart a
        # partir dos arquivos presentes naquele momento, então dois deploys
        # com corpus ligeiramente diferente (Railway vs Kaggle vs Termux)
        # podiam silenciosamente indexar os mesmos caracteres de jeitos
        # diferentes. Aqui o tokenizer vira parte do checkpoint, não do
        # ambiente.
        ckpt_merges = _bpe_merges_from_json(ckpt['bpe_merges'])
        BPE_MERGES = ckpt_merges
        BPE_VOCAB = _bpe_vocab_from_merges(ckpt_merges)
        VOCAB = len(BPE_VOCAB)
        data_tensor = torch.tensor(encode(CORPUS), dtype=torch.long)
        try:
            with open(_TOKENIZER_CACHE_PATH, 'w', encoding='utf-8') as f:
                _json.dump({'vocab_size': VOCAB, 'merges': _bpe_merges_to_json(BPE_MERGES)}, f)
        except Exception:
            pass
    model = TinyAI(ckpt['model_cfg']).to(device)
    ckpt_vocab = ckpt['model_state']['tok_emb.weight'].shape[0]
    # checkpoints salvos antes desse campo existir não têm 'context_len' —
    # nesse caso deduzimos do shape do pos_emb salvo.
    ckpt_context_len = ckpt.get('context_len', ckpt['model_state']['pos_emb.weight'].shape[0])
    vocab_changed = ckpt_vocab != VOCAB
    context_changed = ckpt_context_len != CONTEXT_LEN
    vocab_mismatch = vocab_changed or context_changed
    if legacy_char_ckpt:
        console.print('[bold yellow]⚠️  Checkpoint salvo com o tokenizer char-level antigo — migrando pra BPE.\n'
                       '   → tok_emb/head não são reaproveitáveis entre os dois esquemas de vocab '
                       '(reiniciados do zero, com init aleatório).\n'
                       '   → attention/MLP/MoE (a "estrutura" do modelo) são transplantados normalmente.\n'
                       '   → Pré-treino de recuperação será executado automaticamente.[/bold yellow]')
        _transplant_state_dict(ckpt['model_state'], model, skip_prefixes=('tok_emb.', 'head.'))
        rl_opt = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
        sup_opt = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
        pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.0001)
        vocab_mismatch = True
    elif vocab_mismatch:
        avisos = []
        if vocab_changed:
            d = VOCAB - ckpt_vocab
            avisos.append(f"vocab: checkpoint={ckpt_vocab} tokens, atual={VOCAB} tokens ({d:+d})")
        if context_changed:
            avisos.append(f"context_len: checkpoint={ckpt_context_len}, atual={CONTEXT_LEN}")
        console.print('[bold yellow]⚠️  ' + ' | '.join(avisos) + '[/bold yellow]\n   → Transplantando pesos compatíveis (posições/tokens novos ficam com init aleatório).\n   → Pré-treino de recuperação será executado automaticamente.')
        _transplant_state_dict(ckpt['model_state'], model)
        rl_opt = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
        sup_opt = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
        pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.0001)
    else:
        model.load_state_dict(ckpt['model_state'])
        rl_opt = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
        sup_opt = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
        rl_opt.load_state_dict(ckpt['rl_opt_state'])
        sup_opt.load_state_dict(ckpt['sup_opt_state'])
        pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.0001)
        pt_state = ckpt.get('pretrain_opt_state')
        if pt_state is not None:
            try:
                pretrain_opt.load_state_dict(pt_state)
            except Exception:
                pass
    state['gen'] = ckpt['gen']
    state['temp'] = ckpt['temp']
    state['best_reward'] = ckpt['best_reward']
    state['best_code'] = ckpt['best_code']
    state['best_gen'] = ckpt['best_gen']
    state['config'] = ckpt['config']
    state['mutation_log'] = ckpt['mutation_log']
    state['temp_resets'] = ckpt['temp_resets']
    state['reward_hist'].extend(ckpt.get('reward_hist', []))
    state['stdout_memory'].update(ckpt.get('stdout_memory', []))
    state['stdout_norm_memory'].update(ckpt.get('stdout_norm_mem', []))
    state['n_params'] = model.n_params
    return (model, rl_opt, sup_opt, pretrain_opt, vocab_mismatch)

def get_batch(bs=4):  # era 1 — bs=1 só adicionava ruído de gradiente à toa (um único
                       # exemplo decidindo o step inteiro). bs=4 já reduz bastante a
                       # variância. Não subiu mais que isso porque a atenção aqui é
                       # "ingênua" (sem Flash Attention) e escala O(T²) por camada:
                       # com CONTEXT_LEN=4096, cada unidade de batch custa vários GB
                       # de ativações de atenção. Suba com cautela e monitore RAM/VRAM.
    max_i = len(data_tensor) - CONTEXT_LEN - 1
    if max_i <= 0:
        return (None, None)
    ix = torch.randint(0, max_i, (bs,))
    x = torch.stack([data_tensor[i:i + CONTEXT_LEN] for i in ix])
    y = torch.stack([data_tensor[i + 1:i + CONTEXT_LEN + 1] for i in ix])
    return (x.to(device), y.to(device))

def pretrain(model, steps, lr, status_callback=None, opt=None):
    created_here = opt is None
    if created_here:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    else:
        for pg in opt.param_groups:
            pg['lr'] = lr
    for step in range(steps):
        x, y = get_batch()
        if x is None:
            break
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if status_callback and status_callback(step, steps, loss.item()):
            break  # callback pediu parada (ex: critério de loss)
    return model

def _get_batch_de_texto(texto_encoded: torch.Tensor, bs=1):  # era 16 → 2 → 1, consistente com o treino federado (bs=1)
    max_i = len(texto_encoded) - CONTEXT_LEN - 1
    if max_i <= 0:
        return (None, None)
    ix = torch.randint(0, max_i, (bs,))
    x = torch.stack([texto_encoded[i:i + CONTEXT_LEN] for i in ix])
    y = torch.stack([texto_encoded[i + 1:i + CONTEXT_LEN + 1] for i in ix])
    return (x.to(device), y.to(device))

def set_expert_treinavel(model, expert_id: int, apenas_este=True):
    treinaveis = []
    for name, p in model.named_parameters():
        if not apenas_este:
            p.requires_grad = True
            continue
        alvo = f'.experts.{expert_id}.'
        p.requires_grad = alvo in name
        if p.requires_grad:
            treinaveis.append(name)
    return treinaveis

def treinar_expert(model, expert_id: int, texto: str, steps: int, lr: float, status_callback=None, opt=None, congelar_outros=True):
    cfg = getattr(model, '_cfg', {})
    n_experts = cfg.get('n_experts')
    if not cfg.get('use_moe') or not n_experts:
        raise ValueError('Modelo atual não usa MoE — não há experts pra treinar.')
    if not 0 <= expert_id < n_experts:
        raise ValueError(f'expert_id {expert_id} fora do range (0..{n_experts - 1})')
    tokens = torch.tensor(encode(texto), dtype=torch.long)
    if len(tokens) < CONTEXT_LEN + 2:
        print(f'⚠️  Texto curto demais pra treinar expert {expert_id} ({len(tokens)} tokens, precisa de >= {CONTEXT_LEN + 2}).')
        return None
    if congelar_outros:
        treinaveis = set_expert_treinavel(model, expert_id, apenas_este=True)
        if not treinaveis:
            set_expert_treinavel(model, expert_id, apenas_este=False)
            raise RuntimeError(f'Nenhum parâmetro encontrado pra expert {expert_id} — checa se o modelo é MoE de verdade.')
    created_here = opt is None
    if created_here:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0001)
    else:
        for pg in opt.param_groups:
            pg['lr'] = lr
    try:
        for step in range(steps):
            x, y = _get_batch_de_texto(tokens)
            if x is None:
                break
            _, loss, _ = model(x, y, forced_expert=expert_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            if status_callback:
                status_callback(step, steps, loss.item())
    finally:
        if congelar_outros:
            set_expert_treinavel(model, expert_id, apenas_este=False)
    return model

def novo_estado_inicial():
    # Campos mantidos por compatibilidade com o formato dos checkpoints
    # (save_checkpoint / restore_from_checkpoint) e com o servidor Flask.
    return dict(gen=0, temp=TEMP_START, config=INIT_CONFIG.copy(), n_params=0, best_code='', best_reward=0.0, best_gen=0, reward_hist=deque(maxlen=90), mutation_log=[], stdout_memory=set(), stdout_norm_memory=set(), temp_resets=0)

def _carregar_ou_criar_modelo(state):
    ckpt = load_checkpoint(CHECKPOINT_LAST)
    if ckpt is not None:
        console.print('[bold yellow]♻️  Checkpoint encontrado! Carregando...[/bold yellow]')
        model, rl_opt, sup_opt, pretrain_opt, vocab_mismatch = restore_from_checkpoint(ckpt, state)
        state['last_ckpt_gen'] = state['gen']
        return (model, rl_opt, sup_opt, pretrain_opt)
    console.print('[bold green]🆕 Nenhum checkpoint encontrado. Criando modelo novo.[/bold green]')
    model = TinyAI(INIT_CONFIG).to(device)
    state['n_params'] = model.n_params
    rl_opt = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
    sup_opt = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
    pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.0001)
    return (model, rl_opt, sup_opt, pretrain_opt)
CLI_CATEGORIA_CORPUS = {'code': lambda: CORPUS_CODE, 'general': lambda: CORPUS_GENERAL}
CLI_STEPS_PADRAO = 2000

def _parse_expert_cli(argv):
    args = argv[1:]
    if not args or not args[0].lstrip('-').isdigit():
        return None
    expert_num = int(args[0])
    categoria = None
    steps = None
    it = iter(args[1:])
    for a in it:
        al = a.lower()
        if al in ('--general', '--geral'):
            categoria = 'general'
        elif al in ('--code', '--codigo', '--código'):
            categoria = 'code'
        elif al == '--steps':
            try:
                steps = int(next(it))
            except (StopIteration, ValueError):
                pass
    if categoria is None:
        return None
    return (expert_num - 1, categoria, steps)

def rodar_treino_expert_cli(expert_id: int, categoria: str, steps: int=None):
    steps = steps or CLI_STEPS_PADRAO
    texto = CLI_CATEGORIA_CORPUS[categoria]()
    console.print(f'\n[bold blue]══ 🎯 Treino direcionado de expert (CLI) ══[/bold blue]')
    console.print(f'   expert_id={expert_id}  categoria={categoria}  corpus={len(texto)} chars  steps={steps}\n')
    state = novo_estado_inicial()
    model, rl_opt, sup_opt, pretrain_opt = _carregar_ou_criar_modelo(state)
    n_experts = model._cfg.get('n_experts', 0)
    if not model._cfg.get('use_moe') or expert_id >= n_experts or expert_id < 0:
        console.print(f"[bold red]❌ expert_id {expert_id} inválido pra este modelo (use_moe={model._cfg.get('use_moe')}, n_experts={n_experts}). Lembre: no CLI o número é 1-based (ex: '1' = expert índice 0).[/bold red]\n")
        return

    def cb(step, total, loss):
        if step % 50 == 0 or step == total - 1:
            console.print(f'   step {step}/{total}  loss={loss:.4f}')
    try:
        treinar_expert(model, expert_id, texto, steps, SUPERVISED_LR, status_callback=cb)
    except (ValueError, RuntimeError) as e:
        console.print(f'[bold red]❌ Falha: {e}[/bold red]\n')
        return
    save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
    console.print(f'\n[bold green]✅ Expert {expert_id} ({categoria}) treinado por {steps} steps. Checkpoint salvo em {os.path.abspath(_ckpt_path(CHECKPOINT_LAST))}[/bold green]\n')
    console.print("[dim]Rode 'python ia_server.py' sem argumentos pra continuar o treino geral.[/dim]\n")

def main():
    """Rotina de `python ia_server.py`: treina, salva o checkpoint e encerra."""
    console.print('\n[bold blue]══ 🧬 SELF-EVOLVING AI — Treino (MoE + Checkpoints) ══[/bold blue]')
    console.print(f'   {dev_str}')
    console.print(f'   vocab={VOCAB} tokens (BPE)  |  context={CONTEXT_LEN}  |  corpus={len(data_tensor)} tokens\n')
    if len(data_tensor) <= CONTEXT_LEN + 1:
        console.print('[bold red]❌ Corpus curto demais pra treinar (coloque arquivos .txt/.csv na pasta).[/bold red]\n')
        return
    state = novo_estado_inicial()
    ckpt = load_checkpoint(CHECKPOINT_LAST)
    if ckpt is not None:
        console.print('[bold yellow]♻️  Checkpoint encontrado! Retomando treino...[/bold yellow]')
        model, rl_opt, sup_opt, pretrain_opt, vocab_mismatch = restore_from_checkpoint(ckpt, state)
        if vocab_mismatch:
            steps, lr, label = 2500, PRETRAIN_LR, 'recuperação'
        else:
            steps, lr, label = PRETRAIN_STEPS, PRETRAIN_LR_CONT, 'continuação'
    else:
        console.print('[bold green]🆕 Nenhum checkpoint encontrado. Começando do zero.[/bold green]')
        model = TinyAI(INIT_CONFIG).to(device)
        state['n_params'] = model.n_params
        rl_opt = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
        sup_opt = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
        pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.0001)
        steps, lr, label = PRETRAIN_STEPS, PRETRAIN_LR, 'inicial'
    console.print(f'[bold cyan]🏋️  Treino {label}: {steps} steps  |  lr={lr}  |  params={model.n_params:,}[/bold cyan]\n')
    progresso = {'step': 0, 'loss': None, 'avg': None, 'motivo': 'limite de steps'}
    janela = deque(maxlen=LOSS_WINDOW)
    melhor = {'avg': float('inf'), 'step': 0}

    def cb(step, total, loss):
        n = step + 1
        progresso['step'] = n
        progresso['loss'] = loss
        janela.append(loss)
        avg = sum(janela) / len(janela)
        progresso['avg'] = avg
        if step % 50 == 0 or step == total - 1:
            console.print(f'   step {n}/{total}  loss={loss:.4f}  média({len(janela)})={avg:.4f}')
        if step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
        if n < max(LOSS_MIN_STEPS, LOSS_WINDOW):
            return False
        if avg <= LOSS_TARGET:
            progresso['motivo'] = f'loss alvo atingido (média {avg:.4f} <= {LOSS_TARGET})'
            return True
        if avg < melhor['avg'] - LOSS_PLATEAU_MIN_DELTA:
            melhor['avg'], melhor['step'] = avg, n
        elif n - melhor['step'] >= LOSS_PLATEAU_PATIENCE:
            progresso['motivo'] = f'platô (média sem cair {LOSS_PLATEAU_MIN_DELTA} há {LOSS_PLATEAU_PATIENCE} steps; melhor {melhor["avg"]:.4f})'
            return True
        return False
    model.train()
    try:
        pretrain(model, steps, lr, cb, opt=pretrain_opt)
    except KeyboardInterrupt:
        progresso['motivo'] = 'interrupção manual / --max-minutes'
        console.print('\n[yellow]⏹️  Interrompido — salvando o que já foi treinado...[/yellow]')
    console.print('\n[yellow]💾 Salvando checkpoint final...[/yellow]')
    save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
    console.print('\n[bold blue]══ RESUMO FINAL ══[/bold blue]')
    console.print(f"   Steps rodados: {progresso['step']}/{steps}")
    if progresso['loss'] is not None:
        console.print(f"   Loss final: {progresso['loss']:.4f}  (média móvel: {progresso['avg']:.4f})")
    console.print(f"   Parou por: {progresso['motivo']}")
    console.print(f"   Arquitetura: {state['config']}")
    console.print(f"   Parâmetros: {state['n_params']:,}")
    console.print(f'   Checkpoint em: [bold]{os.path.abspath(_ckpt_path(CHECKPOINT_LAST))}[/bold]\n')
    console.print('[bold green]✅ Treino concluído. Encerrando.[/bold green]\n')
def _truncate_kv_cache(past_kv, max_len: int):
    """Corta o cache K/V pra caber no teto de CONTEXT_LEN — sem isso, uma
    geração longa o suficiente estoura o índice do pos_emb (que só tem
    CONTEXT_LEN posições) e quebra com erro de shape."""
    if past_kv is None:
        return None
    out = []
    for k, v in past_kv:
        if k.shape[2] > max_len:
            k = k[:, :, -max_len:, :].contiguous()
            v = v[:, :, -max_len:, :].contiguous()
        out.append((k, v))
    return out


@torch.no_grad()
def generate_text(model, prompt: str, max_new: int=200, temperature: float=0.8) -> str:
    model.eval()
    tokens = encode(prompt)
    if not tokens:
        tokens = [NEWLINE_ID]
    # deixa 1 posição de folga pro próximo token gerado não estourar CONTEXT_LEN
    tok = torch.tensor([tokens[-(CONTEXT_LEN - 1):]], dtype=torch.long, device=device)
    out_ids = []
    logits, _, past_kv = model(tok, use_cache=True)
    next_logits = logits[:, -1, :] / max(temperature, 0.1)
    for _ in range(max_new):
        probs = F.softmax(next_logits, dim=-1)
        sampled = torch.distributions.Categorical(probs).sample()
        tid = sampled.item()
        if tid == EOS_ID:
            break
        out_ids.append(tid)
        tok = sampled.view(1, 1)
        past_kv = _truncate_kv_cache(past_kv, CONTEXT_LEN - 1)
        logits, _, past_kv = model(tok, past_kv=past_kv, use_cache=True)
        next_logits = logits[:, -1, :] / max(temperature, 0.1)
    model.train()
    return decode(out_ids)


_WIKI_STOP_SECTIONS = {
    'referencias', 'referências', 'ver tambem', 'ver também', 'ligacoes externas',
    'ligações externas', 'bibliografia', 'notas', 'fontes', 'leitura adicional',
    'leitura complementar', 'obras citadas', 'anexos', 'ver também.',
}

def _limpar_texto_wiki(texto: str) -> str:
    """Remove cabeçalhos de seção ('== Título ==') do extract da Wikipedia e
    corta o artigo assim que bate numa seção de referências/links (que no
    modo explaintext vira uma lista de citações truncadas, não prosa)."""
    if not texto:
        return texto
    linhas_limpas = []
    for linha in texto.split('\n'):
        m = _re.match('^(=+)\\s*(.+?)\\s*\\1$', linha.strip())
        if m:
            titulo_norm = _re.sub('\\s+', ' ', m.group(2).strip().lower())
            if titulo_norm in _WIKI_STOP_SECTIONS:
                break
            continue  # remove o cabeçalho (ex: '== História ==') mas segue lendo
        linhas_limpas.append(linha)
    texto_limpo = '\n'.join(linhas_limpas)
    texto_limpo = _re.sub('\n{3,}', '\n\n', texto_limpo)
    return texto_limpo.strip()

def _wiki_random_page_text(lang: str='en'):
    if requests is None:
        raise RuntimeError("Pacote 'requests' não instalado. Rode: pip install requests")
    api_url = f'https://{lang}.wikipedia.org/w/api.php'
    headers = {'User-Agent': 'SelfEvolvingAI/1.0 (contato@exemplo.com)'}
    r = requests.get(api_url, params={'action': 'query', 'list': 'random', 'rnnamespace': 0, 'rnlimit': 1, 'format': 'json'}, headers=headers, timeout=10)
    title = r.json()['query']['random'][0]['title']
    r2 = requests.get(api_url, params={'action': 'query', 'prop': 'extracts', 'explaintext': True, 'titles': title, 'format': 'json'}, headers=headers, timeout=10)
    pages = r2.json()['query']['pages']
    page = next(iter(pages.values()))
    texto_bruto = page.get('extract', '')
    return (title, _limpar_texto_wiki(texto_bruto))


# Prefetch simples de páginas da Wikipedia — o servidor SÓ busca e guarda o
# texto pra distribuir nos jobs; quem treina com isso são os voluntários.
_wiki_cache_lock = threading.Lock()
_wiki_cache: list = []       # fila de (title, texto) já buscados, prontos pra usar
_WIKI_CACHE_TARGET = 10       # quantas páginas manter prontas no buffer
_WIKI_MODE = False            # setado por run_server(wiki=True)
WIKI_CORPUS_DIR = './wiki_corpus'  # cada página vira um .txt aqui — substrato pra RAG/replay no chat.py


def _slug(title: str) -> str:
    safe = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in title).strip()
    return (safe or 'pagina')[:120]


def _save_wiki_page_to_disk(title: str, texto: str):
    """Persiste a página em disco — sem isso, o conteúdo buscado só existe
    na RAM do servidor e some a cada restart/redeploy. Um arquivo por página
    (nome = título) pra dar pra inspecionar/filtrar manualmente depois, e
    pro chat.py usar como pool de replay (--wiki-corpus-dir)."""
    try:
        os.makedirs(WIKI_CORPUS_DIR, exist_ok=True)
        path = os.path.join(WIKI_CORPUS_DIR, _slug(title) + '.txt')
        if not os.path.exists(path):  # não reescreve se já tem (evita duplicar I/O em páginas repetidas)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(texto)
    except OSError as e:
        console.print(f'[bold yellow]⚠️  Não consegui salvar a página em disco: {e}[/bold yellow]')


def _expand_vocab_with_text(model: 'TinyAI', texto: str) -> bool:
    """No char-level antigo, essa função crescia o vocab (chars/VOCAB/c2i/i2c)
    toda vez que a Wikipedia trazia um caractere nunca visto, redimensionando
    tok_emb/head em memória. Com BPE byte-level isso deixou de ser
    necessário: os 256 bytes-base já cobrem qualquer texto UTF-8 por
    construção, então não existe mais "caractere fora do vocab". Mantida
    como no-op só pra não quebrar quem chama (_wiki_prefetch_loop). Se um dia
    fizer sentido retreinar os merges do BPE periodicamente com texto novo,
    é aqui que essa lógica entraria — mas isso implica reconstruir
    tok_emb/head do zero (o índice de cada token muda), não redimensionar."""
    return False


def _wiki_prefetch_loop(model: 'TinyAI'):
    console.print('[bold cyan]📖 Prefetch de conteúdo da Wikipedia iniciado (--federated --wiki)[/bold cyan]')
    while True:
        with _wiki_cache_lock:
            precisa = len(_wiki_cache) < _WIKI_CACHE_TARGET
        if not precisa:
            time.sleep(2)
            continue
        try:
            title, texto = _wiki_random_page_text()
        except Exception as e:
            console.print(f'[bold red]⚠️  Falha ao buscar página da Wikipedia: {e}[/bold red]')
            time.sleep(5)
            continue
        if len(texto) <= CONTEXT_LEN + 1:
            continue
        _expand_vocab_with_text(model, texto)
        _save_wiki_page_to_disk(title, texto)
        with _wiki_cache_lock:
            _wiki_cache.append((title, texto))
        console.print(f"[cyan]📖 Wiki em cache: '{title}' ({len(texto)} chars) — buffer: {len(_wiki_cache)}/{_WIKI_CACHE_TARGET}[/cyan]")


def _wiki_take_page():
    """Pega uma página pronta do cache pra mandar num job. Se o buffer estiver
    vazio (raro, prefetch ainda não alcançou), busca uma na hora (bloqueia
    esse request, mas não trava o servidor todo — sem lock de treino aqui)."""
    with _wiki_cache_lock:
        if _wiki_cache:
            return _wiki_cache.pop(0)
    title, texto = _wiki_random_page_text()
    return title, texto


def _build_flask_app(model, rl_opt, sup_opt, pretrain_opt, state, federated: bool=False):
    app = Flask(__name__)


    @app.route('/v1/model', methods=['GET'])
    def download_model():
        with _server_lock:
            save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
            path = _ckpt_path(CHECKPOINT_LAST)
        return send_file(path, as_attachment=True, download_name='checkpoint_last.pt')

    @app.route('/v1/chat', methods=['POST'])
    def chat():
        data = request.get_json(force=True, silent=True) or {}
        prompt = data.get('prompt') or data.get('message') or ''
        max_new = int(data.get('max_tokens', 200))
        temperature = float(data.get('temperature', 0.8))
        with _server_lock:
            reply = generate_text(model, prompt, max_new=max_new, temperature=temperature)
        return jsonify({'id': 'chatcmpl-local', 'object': 'chat.completion', 'model': 'tinyai-local', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': reply}, 'finish_reason': 'stop'}]})

    @app.route('/v1/status', methods=['GET'])
    def status():
        with _server_lock:
            payload = {'gen': state.get('gen', 0), 'n_params': state.get('n_params'), 'vocab_size': VOCAB, 'config': model._cfg, 'federated_enabled': federated}
            if federated:
                payload['federated_version'] = _fed_state['version']
                payload['federated_stats'] = _fed_state['stats']
            return jsonify(payload)

    if federated:
        @app.route('/v1/job', methods=['GET'])
        def get_job():
            own_content = request.args.get('own_content', '0') == '1'
            with _server_lock:
                payload = _fed_issue_job(model, own_content)
            return jsonify(payload)

        @app.route('/v1/submit', methods=['POST'])
        def submit_job():
            data = request.get_json(force=True, silent=True) or {}
            job_id = data.get('job_id')
            version = data.get('version')
            n_steps = data.get('n_steps')
            delta_b64 = data.get('delta')
            worker_id = str(data.get('worker_id', 'anonimo'))[:64]
            if not job_id or version is None or not n_steps or not delta_b64:
                return jsonify({'accepted': False, 'reason': 'campos faltando (job_id, version, n_steps, delta)'}), 400
            with _server_lock:
                result = _fed_submit_delta(model, job_id, int(version), int(n_steps), delta_b64, worker_id)
                if result.get('aggregated'):
                    state['gen'] = state.get('gen', 0) + 1
                    save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
            return jsonify(result)

    return app


def _sample_batch_local(encoded: torch.Tensor, context_len: int, dev):
    if len(encoded) <= context_len + 1:
        return None, None
    i = torch.randint(0, len(encoded) - context_len - 1, (1,)).item()
    x = encoded[i:i + context_len].unsqueeze(0).to(dev)
    y = encoded[i + 1:i + context_len + 1].unsqueeze(0).to(dev)
    return x, y


# Páginas recentes já treinadas (título, tensor codificado) — usado pra
# misturar exemplos "antigos" no batch de treino da página atual, em vez de
# treinar isolado nela. Sem isso, cada página nova ia sobrescrevendo o que
# foi aprendido nas anteriores (é o que causava o modelo "grudar" no último
# assunto visto, não importa o prompt).
_replay_buffer: deque = deque(maxlen=REPLAY_BUFFER_MAX_PAGES)


def _sample_batch_misto(encoded_atual: torch.Tensor, buffer_replay: list, context_len: int, dev,
                         bs: int = REPLAY_BATCH_SIZE, fracao_replay: float = REPLAY_FRACTION):
    """Monta um batch misturando janelas da página atual com janelas de
    páginas antigas do buffer de replay. Sempre garante pelo menos 1 exemplo
    da página atual (senão não haveria treino nela)."""
    n_replay = min(int(round(bs * fracao_replay)), bs - 1) if buffer_replay else 0
    n_atual = bs - n_replay
    encodeds = [encoded_atual] * n_atual + [random.choice(buffer_replay)[1] for _ in range(n_replay)]
    xs, ys = [], []
    for enc in encodeds:
        if len(enc) <= context_len + 1:
            continue
        i = torch.randint(0, len(enc) - context_len - 1, (1,)).item()
        xs.append(enc[i:i + context_len])
        ys.append(enc[i + 1:i + context_len + 1])
    if not xs:
        return None, None
    return torch.stack(xs).to(dev), torch.stack(ys).to(dev)


def _local_training_direct(model: 'TinyAI', pretrain_opt, rl_opt, sup_opt, state: dict,
                            loss_target: float = LOCAL_TRAIN_LOSS_TARGET,
                            max_steps_por_pagina: int = LOCAL_TRAIN_MAX_STEPS_POR_PAGINA):
    """Treino local DIRETO no modelo mestre — sem cópia extra, sem
    compressão de pesos (isso é o que pesava em RAM no modo via job
    federado, que existe pra simular um voluntário de verdade isolado).
    Consome as páginas do mesmo buffer de prefetch que o --wiki alimenta."""
    console.print("[bold green]💻 Treino local direto iniciado (--training-local, sem federação/duplicação de modelo)[/bold green]")
    local_vocab_epoch = _VOCAB_EPOCH
    while True:
        with _wiki_cache_lock:
            pagina = _wiki_cache.pop(0) if _wiki_cache else None
        if pagina is None:
            time.sleep(1)
            continue
        title, texto = pagina
        encoded = torch.tensor(encode(texto), dtype=torch.long)
        if len(encoded) <= CONTEXT_LEN + 1:
            continue
        console.print(f"[cyan]💻 Treinando local: '{title}' ({len(texto)} chars)[/cyan]")
        step = 0
        last_loss = None
        while step < max_steps_por_pagina:
            if _VOCAB_EPOCH != local_vocab_epoch:
                # vocab mudou (tok_emb/head foram trocados) — o optimizer antigo
                # ainda referencia os tensores órfãos; reconstruir evita vazar
                # memória a cada expansão E garante que os pesos novos recebam
                # atualização de verdade (senão ficam congelados pra sempre)
                with _server_lock:
                    pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.0001)
                    rl_opt = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
                    sup_opt = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
                    local_vocab_epoch = _VOCAB_EPOCH
                console.print("[cyan]🔄 Vocab mudou — optimizer reconstruído (evita vazamento de memória)[/cyan]")
            x, y = _sample_batch_misto(encoded, list(_replay_buffer), CONTEXT_LEN, device,
                                        fracao_replay=REPLAY_FRACTION if _replay_buffer else 0.0)
            if x is None:
                break
            with _server_lock:
                _, loss, _ = model(x, y)
                pretrain_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                pretrain_opt.step()
            last_loss = loss.item()
            step += 1
            if step == 1 or step % 1 == 0:
                console.print(f'   step {step}  loss={last_loss:.4f}')
            if last_loss <= loss_target:
                break
        with _server_lock:
            state['gen'] = state.get('gen', 0) + 1
            save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
        _replay_buffer.append((title, encoded))
        motivo = 'atingiu loss alvo' if last_loss is not None and last_loss <= loss_target else 'limite de steps'
        console.print(f"[bold green]✅ '{title}' concluída ({motivo}) — loss final={last_loss:.4f}, {step} steps. "
                       f"Checkpoint salvo. (replay buffer: {len(_replay_buffer)} páginas)[/bold green]")


def run_server(wiki: bool=False, host: str='0.0.0.0', port: int=5000, federated: bool=False, training_local: bool=False):
    global _WIKI_MODE
    if Flask is None:
        console.print('[bold red]❌ Flask não instalado. Rode: pip install flask[/bold red]')
        return
    console.print('\n[bold blue]══ 🧬 SELF-EVOLVING AI — Servidor Flask ══[/bold blue]')
    state = novo_estado_inicial()
    model, rl_opt, sup_opt, pretrain_opt = _carregar_ou_criar_modelo(state)
    if wiki:
        _WIKI_MODE = True
        t = threading.Thread(target=_wiki_prefetch_loop, args=(model,), daemon=True)
        t.start()
    if training_local:
        if not wiki:
            console.print('[bold yellow]⚠️  --training-local consome páginas do buffer da Wikipedia — ligando --wiki automaticamente.[/bold yellow]')
            _WIKI_MODE = True
            t = threading.Thread(target=_wiki_prefetch_loop, args=(model,), daemon=True)
            t.start()
        # deixa pelo menos 1 núcleo de folga pro Flask não ficar starved
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        except Exception:
            pass
        t2 = threading.Thread(target=_local_training_direct, args=(model, pretrain_opt, rl_opt, sup_opt, state), daemon=True)
        t2.start()
    app = _build_flask_app(model, rl_opt, sup_opt, pretrain_opt, state, federated=federated)
    rotas = 'GET /v1/model, POST /v1/chat, GET /v1/status'
    if federated:
        rotas += ', GET /v1/job, POST /v1/submit'
    fonte = 'Wikipedia (prefetch)' if wiki else 'corpus local'
    extra = ' + treino local direto ligado' if training_local else ''
    console.print(f'[bold green]🚀 Servidor em http://{host}:{port}  ({rotas})  — conteúdo dos jobs: {fonte}{extra}[/bold green]\n')
    app.run(host=host, port=port, threaded=True)


def _setup_time_limit(argv):
    """Configura um alarme para interromper o processo graciosamente após
    N minutos, disparando um KeyboardInterrupt no thread principal.
    O loop de treino já trata KeyboardInterrupt salvando o checkpoint final
    e encerrando sozinho — então isso resolve o problema do Kaggle nunca
    considerar o processo 'terminado' em modos que rodam para sempre.
    Uso: python ia_server.py --max-minutes 120
    (funciona também combinado com --serve, --wiki, --training-local etc.)
    """
    if '--max-minutes' not in argv:
        return
    try:
        minutos = float(argv[argv.index('--max-minutes') + 1])
    except (ValueError, IndexError):
        console.print('[bold red]❌ --max-minutes precisa de um número (ex: --max-minutes 120)[/bold red]')
        return

    def _time_up(signum, frame):
        console.print(f'\n[bold yellow]⏰ Tempo limite de {minutos:.0f} min atingido — encerrando graciosamente...[/bold yellow]')
        raise KeyboardInterrupt

    signal.signal(signal.SIGALRM, _time_up)
    signal.alarm(int(minutos * 60))
    console.print(f'[bold blue]⏱️  Auto-encerramento configurado para {minutos:.0f} minutos[/bold blue]')


if __name__ == '__main__':
    _setup_time_limit(sys.argv)
    if '--serve' in sys.argv:
        _port = 5000
        if '--port' in sys.argv:
            try:
                _port = int(sys.argv[sys.argv.index('--port') + 1])
            except (ValueError, IndexError):
                pass
        run_server(wiki='--wiki' in sys.argv, port=_port, federated='--federated' in sys.argv, training_local='--training-local' in sys.argv)
    else:
        _cli = _parse_expert_cli(sys.argv)
        if _cli is not None:
            _expert_id, _categoria, _steps = _cli
            rodar_treino_expert_cli(_expert_id, _categoria, _steps)
        else:
            main()