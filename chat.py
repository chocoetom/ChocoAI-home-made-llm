#!/usr/bin/env python3
"""
chat.py — chatbot de terminal para conversar com um checkpoint gerado pelo
ia_server.py (TinyAI denso + Chain-of-Thought).

Por que este arquivo não importa ia_server.py diretamente:
    Importar ia_server.py dispara a leitura e tokenização do corpus de treino
    inteiro e outras coisas que só fazem sentido durante o treino. Para só
    conversar com o modelo já treinado isso é trabalho perdido. Por isso a
    arquitetura e o tokenizer estão reimplementados aqui (cópia fiel do
    ia_server.py, CKPT_FORMAT 2). Pesos e merges do BPE vêm do checkpoint.

Níveis de raciocínio (o que muda em cada um):
    off    conversa direta (<|chat|>), com histórico, sem pensar
    baixo  pensa pouco: até 64 tokens de raciocínio, uma tentativa
    medio  pensa com calma: até 160 tokens, uma tentativa (padrão)
    alto   pensa muito: até 256 tokens e 5 tentativas independentes;
           a resposta final é a mais votada (self-consistency)

Uso:
    python chat.py
    python chat.py --ckpt checkpoints/checkpoint_last.pt --raciocinio alto
    python chat.py --max-tokens 300 --temperature 0.9
    python chat.py --cpu
    python chat.py --esconder-pensamento     # não mostra o raciocínio

Comandos dentro do chat:
    /sair, /exit             encerra
    /reset                   limpa o histórico
    /raciocinio <nivel>      off | baixo | medio | alto   (atalho: /r)
    /pensar                  mostra/esconde o raciocínio
    /temp 0.9                temperatura do modo chat (off)
    /max 300                 máximo de tokens da resposta
"""
import argparse
import codecs
import heapq
import io
import math
import os
import re
import sys
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()

# ═══════════════════════════════════════════════════════════════════════
# TOKENS ESPECIAIS (mesma ordem do ia_server.py — a ordem define os ids)
# ═══════════════════════════════════════════════════════════════════════
Q_TOKEN = '<|q|>'
THINK_TOKEN = '<|think|>'
ANSWER_TOKEN = '<|answer|>'
CHAT_TOKEN = '<|chat|>'
COT_TOKEN = '<|cot|>'
USER_TOKEN = '<|user|>'
ASSISTANT_TOKEN = '<|assistant|>'
SPECIAL_TOKENS = (Q_TOKEN, THINK_TOKEN, ANSWER_TOKEN, CHAT_TOKEN, COT_TOKEN, USER_TOKEN, ASSISTANT_TOKEN)
_SPECIAL_RE = re.compile('(' + '|'.join(re.escape(t) for t in SPECIAL_TOKENS) + ')')

EOS = '\x00'
EOS_ID = 0  # byte 0 — nunca sofre merge
GEN_TOP_K = 40
CKPT_FORMAT = 2

# ═══════════════════════════════════════════════════════════════════════
# NÍVEIS DE RACIOCÍNIO
#   max_think : teto de tokens de pensamento (se estourar, o <|answer|> é forçado)
#   temp      : temperatura da geração (menor = mais determinístico)
#   votos     : nº de tentativas independentes; >1 => vence a resposta mais votada
# ═══════════════════════════════════════════════════════════════════════
NIVEIS = {
    'off':   dict(max_think=0,   temp=None, votos=1, desc='conversa direta, sem pensar'),
    'baixo': dict(max_think=64,  temp=0.5,  votos=1, desc='raciocínio curto'),
    'medio': dict(max_think=160, temp=0.4,  votos=1, desc='raciocínio normal'),
    'alto':  dict(max_think=256, temp=0.6,  votos=5, desc='raciocínio longo + votação entre 5 tentativas'),
}
_ALIASES_NIVEL = {
    '0': 'off', 'nenhum': 'off', 'desligado': 'off',
    '1': 'baixo', 'low': 'baixo', 'l': 'baixo',
    '2': 'medio', 'médio': 'medio', 'med': 'medio', 'medium': 'medio', 'm': 'medio',
    '3': 'alto', 'high': 'alto', 'h': 'alto',
}


def _normalizar_nivel(s: str):
    s = (s or '').strip().lower()
    s = _ALIASES_NIVEL.get(s, s)
    return s if s in NIVEIS else None


# ═══════════════════════════════════════════════════════════════════════
# TOKENIZER BPE byte-level — só encode/decode, os merges vêm do checkpoint
# ═══════════════════════════════════════════════════════════════════════

def _bpe_vocab_from_merges(merges: dict) -> dict:
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in sorted(merges.items(), key=lambda kv: kv[1]):
        vocab[idx] = vocab[a] + vocab[b]
    return vocab


def _bpe_merges_from_json(data: list) -> dict:
    return {tuple(p): i for p, i in data}


def _bpe_encode(text: str, merges: dict) -> list:
    ids = list(text.encode('utf-8'))
    n = len(ids)
    if n < 2 or not merges:
        return ids
    nxt = list(range(1, n)) + [-1]
    prv = list(range(-1, n - 1))
    alive = [True] * n
    heap = []
    for i in range(n - 1):
        rank = merges.get((ids[i], ids[i + 1]))
        if rank is not None:
            heapq.heappush(heap, (rank, i))
    while heap:
        rank, i = heapq.heappop(heap)
        if not alive[i]:
            continue
        j = nxt[i]
        if j == -1:
            continue
        if merges.get((ids[i], ids[j])) != rank:
            continue
        p_, n_ = (prv[i], nxt[j])
        ids[i] = rank
        alive[j] = False
        nxt[i] = n_
        if n_ != -1:
            prv[n_] = i
        if p_ != -1:
            newrank = merges.get((ids[p_], ids[i]))
            if newrank is not None:
                heapq.heappush(heap, (newrank, p_))
        if n_ != -1:
            newrank = merges.get((ids[i], ids[n_]))
            if newrank is not None:
                heapq.heappush(heap, (newrank, i))
    out = []
    i = 0
    while i != -1:
        if alive[i]:
            out.append(ids[i])
        i = nxt[i]
    return out


class Tokenizer:
    """BPE + tokens especiais atômicos (ids logo depois do vocab do BPE)."""

    def __init__(self, merges: dict):
        self.merges = merges
        self.vocab = _bpe_vocab_from_merges(merges)
        self.special_base = len(self.vocab)
        self.special = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.vocab[self.special_base + i] = tok.encode('utf-8')
            self.special[tok] = self.special_base + i
        self.size = len(self.vocab)
        self.Q, self.THINK, self.ANSWER = (self.special[Q_TOKEN], self.special[THINK_TOKEN], self.special[ANSWER_TOKEN])
        self.CHAT, self.COT = (self.special[CHAT_TOKEN], self.special[COT_TOKEN])
        self.USER, self.ASSISTANT = (self.special[USER_TOKEN], self.special[ASSISTANT_TOKEN])

    def encode(self, s: str) -> list:
        out = []
        for parte in _SPECIAL_RE.split(s):
            if not parte:
                continue
            sid = self.special.get(parte)
            if sid is not None:
                out.append(sid)
            else:
                out.extend(_bpe_encode(parte, self.merges))
        return out

    def piece(self, tid: int) -> bytes:
        return self.vocab.get(tid, b'?')


def _limpar_entrada(s: str) -> str:
    """Tira tokens especiais e EOS do que o usuário digita (não dá pra forjar a estrutura)."""
    return _SPECIAL_RE.sub('', s or '').replace(EOS, '')


# ═══════════════════════════════════════════════════════════════════════
# ARQUITETURA DO MODELO (cópia fiel do ia_server.py — denso, sem MoE)
# ═══════════════════════════════════════════════════════════════════════
VOCAB = None
CONTEXT_LEN = None


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.hd = embed_dim // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rd = nn.Dropout(dropout)

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
        p = self.dropout if self.training else 0.0
        if past_kv is None:
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        elif T == 1:
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        else:
            Tk = k.shape[2]
            mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        return (self.rd(self.out(o)), new_kv)


class MLP(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.act = nn.GELU()
        self.proj = nn.Linear(4 * embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return (x, new_kv)


class TinyAI(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ed, nh, nl, do = (cfg['embed_dim'], cfg['n_heads'], cfg['n_layers'], cfg['dropout'])
        self.tok_emb = nn.Embedding(VOCAB, ed)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, ed)
        self.drop = nn.Dropout(do)
        self.blocks = nn.ModuleList([Block(ed, nh, do) for _ in range(nl)])
        self.ln_f = nn.LayerNorm(ed)
        self.head = nn.Linear(ed, VOCAB, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying (igual ao ia_server.py)
        self._cfg = cfg

    def forward(self, idx, targets=None, past_kv=None, use_cache=False, only_last=False):
        B, T = idx.shape
        past_len = past_kv[0][0].shape[2] if past_kv is not None else 0
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pkv = past_kv[i] if past_kv is not None else None
            x, nkv = block(x, past_kv=pkv, use_cache=use_cache)
            if use_cache:
                new_kvs.append(nkv)
        x = self.ln_f(x)
        if only_last:
            x = x[:, -1:, :]
        return (self.head(x), None, new_kvs)

    @property
    def n_params(self):
        return sum((p.numel() for p in self.parameters()))


def _truncate_kv_cache(past_kv, max_len: int):
    if past_kv is None or max_len <= 0:
        return None
    out = []
    for k, v in past_kv:
        if k.shape[2] > max_len:
            k = k[:, :, -max_len:, :].contiguous()
            v = v[:, :, -max_len:, :].contiguous()
        out.append((k, v))
    return out


# ═══════════════════════════════════════════════════════════════════════
# CARREGAMENTO — com feedback visual em cada etapa
# ═══════════════════════════════════════════════════════════════════════

def _default_checkpoint_path() -> str:
    base = '/kaggle/working/checkpoints' if os.path.isdir('/kaggle/working') else './checkpoints'
    return os.path.join(base, 'checkpoint_last.pt')


def _ler_checkpoint_com_progresso(path: str, device: torch.device) -> dict:
    """Lê o arquivo em pedaços com barra de progresso real (bytes lidos do disco)."""
    tamanho = os.path.getsize(path)
    buf = io.BytesIO()
    chunk = 4 * 1024 * 1024
    with Progress(
        '[bold cyan]📦 Carregando checkpoint[/bold cyan]',
        BarColumn(),
        '[progress.percentage]{task.percentage:>3.0f}%',
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task('carregando', total=tamanho)
        with open(path, 'rb') as f:
            while True:
                data = f.read(chunk)
                if not data:
                    break
                buf.write(data)
                progress.update(task, advance=len(data))
    buf.seek(0)
    with console.status('[bold cyan]📤 Deserializando tensores (torch.load)...[/bold cyan]', spinner='dots'):
        ckpt = torch.load(buf, map_location=device, weights_only=False)
    return ckpt


def carregar_modelo(ckpt_path: str, force_cpu: bool = False):
    global VOCAB, CONTEXT_LEN

    if not os.path.exists(ckpt_path):
        console.print(f'[bold red]❌ Checkpoint não encontrado: {ckpt_path}[/bold red]')
        console.print('   Rode o treino primeiro (ia_server.py) ou passe --ckpt apontando pro arquivo certo.')
        sys.exit(1)

    device = torch.device('cpu' if (force_cpu or not torch.cuda.is_available()) else 'cuda')
    ckpt = _ler_checkpoint_com_progresso(ckpt_path, device)

    if ckpt.get('format') != CKPT_FORMAT:
        console.print('[bold red]❌ Checkpoint em formato antigo (MoE).[/bold red]')
        console.print('   Este chat.py só abre checkpoints do ia_server.py denso + CoT (format 2). '
                      'Treine de novo com o ia_server.py atual.')
        sys.exit(1)

    with console.status('[bold cyan]🔤 Reconstruindo tokenizer BPE a partir do checkpoint...[/bold cyan]', spinner='dots'):
        tk = Tokenizer(_bpe_merges_from_json(ckpt['bpe_merges']))
        VOCAB = ckpt.get('vocab_size', tk.size)
        CONTEXT_LEN = ckpt.get('context_len', ckpt['model_state']['pos_emb.weight'].shape[0])
        if VOCAB != tk.size:
            console.print(f'[yellow]⚠️  vocab_size do checkpoint ({VOCAB}) difere do tokenizer ({tk.size}).[/yellow]')

    with console.status('[bold cyan]🧠 Construindo arquitetura do modelo...[/bold cyan]', spinner='dots'):
        model = TinyAI(ckpt['model_cfg']).to(device)

    with console.status('[bold cyan]⚙️  Carregando pesos treinados no modelo...[/bold cyan]', spinner='dots'):
        # export_weights() (fp16) não guarda head.weight: ele é o mesmo tensor de tok_emb
        res = model.load_state_dict(
            {k: (v.float() if v.is_floating_point() else v) for k, v in ckpt['model_state'].items()},
            strict=False,
        )
        faltando = [k for k in res.missing_keys if k != 'head.weight']
        if faltando or res.unexpected_keys:
            console.print(f'[bold red]❌ Pesos incompatíveis. Faltando: {faltando[:5]} · Sobrando: {res.unexpected_keys[:5]}[/bold red]')
            sys.exit(1)
        model.eval()

    resumo = (
        f"Parâmetros: [bold]{model.n_params:,}[/bold]\n"
        f"Vocabulário: [bold]{VOCAB}[/bold] tokens (BPE + {len(SPECIAL_TOKENS)} especiais)\n"
        f"Contexto: [bold]{CONTEXT_LEN}[/bold] tokens\n"
        f"Geração salva no checkpoint: [bold]{ckpt.get('gen', '?')}[/bold]\n"
        f"Device: [bold]{device}[/bold]"
    )
    console.print(Panel(resumo, title='🧬 Modelo carregado', border_style='green'))
    return model, tk, device


# ═══════════════════════════════════════════════════════════════════════
# GERAÇÃO — chat direto e raciocínio (CoT) com níveis
# ═══════════════════════════════════════════════════════════════════════

def _sample(logits, temperature: float, top_k: int = GEN_TOP_K, ban: tuple = ()) -> int:
    logits = logits.float() / max(temperature, 0.1)
    if ban:
        logits[:, list(ban)] = float('-inf')
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


class Gerador:
    """Estado de uma geração (logits + cache K/V) e as fases de geração."""

    def __init__(self, model, tk: Tokenizer, device):
        self.model, self.tk, self.device = model, tk, device
        self.logits = None
        self.past = None
        self.parou_em = None  # id do token que encerrou a última fase (ou None se estourou o teto)

    def _alimentar(self, ids: list):
        tok = torch.tensor([ids], dtype=torch.long, device=self.device)
        self.past = _truncate_kv_cache(self.past, CONTEXT_LEN - len(ids))
        logits, _, self.past = self.model(tok, past_kv=self.past, use_cache=True, only_last=True)
        self.logits = logits[:, -1, :]

    def iniciar(self, ids: list):
        self.past = None
        self._alimentar(ids)

    def _fase(self, max_new: int, temperature: float, ban: tuple, stop: tuple):
        """Gera ids até um token de `stop` ou até max_new. Guarda em self.parou_em quem parou."""
        self.parou_em = None
        for _ in range(max_new):
            tid = _sample(self.logits, temperature, ban=ban)
            if tid in stop:
                self.parou_em = tid
                return
            yield tid
            self._alimentar([tid])

    def _textos(self, ids_iter):
        """ids -> pedaços de texto, com decoder UTF-8 incremental (evita '�' no meio de um caractere)."""
        inc = codecs.getincrementaldecoder('utf-8')(errors='replace')
        for tid in ids_iter:
            t = inc.decode(self.tk.piece(tid))
            if t:
                yield t
        t = inc.decode(b'', final=True)
        if t:
            yield t

    # --- modo chat direto (nível "off") -------------------------------------------------
    @torch.no_grad()
    def chat(self, mensagem: str, historico: list, max_new: int, temperature: float):
        tk = self.tk
        mensagem = _limpar_entrada(mensagem)
        historico = [(_limpar_entrada(u), _limpar_entrada(a)) for u, a in historico]
        limite = CONTEXT_LEN - 8
        while True:
            partes = [CHAT_TOKEN]
            for u, a in historico:
                partes.append(f'{USER_TOKEN}{u.strip()}\n{ASSISTANT_TOKEN}{a.strip()}\n')
            partes.append(f'{USER_TOKEN}{mensagem.strip()}\n{ASSISTANT_TOKEN}')
            ids = tk.encode(''.join(partes))
            if len(ids) <= limite or not historico:
                break
            historico = historico[1:]  # descarta o turno mais antigo inteiro
        if len(ids) > limite:
            ids = [tk.CHAT] + ids[-(limite - 1):]
        self.iniciar(ids)
        ban = (tk.Q, tk.THINK, tk.ANSWER, tk.CHAT, tk.COT)
        yield from self._textos(self._fase(max_new, temperature, ban, (EOS_ID, tk.USER, tk.ASSISTANT)))

    # --- modo raciocínio (níveis baixo/medio/alto) --------------------------------------
    @torch.no_grad()
    def raciocinar(self, pergunta: str, max_think: int, max_answer: int, temperature: float):
        """Gera (fase, texto) com fase in {'think', 'answer'}.
        Se o pensamento estourar max_think sem o modelo emitir <|answer|>, ele é forçado."""
        tk = self.tk
        corpo = tk.encode(_limpar_entrada(pergunta).strip() + '\n')[-(CONTEXT_LEN - 8):]
        self.iniciar([tk.COT, tk.Q] + corpo + [tk.THINK])
        estruturais = (tk.Q, tk.THINK, tk.CHAT, tk.COT, tk.USER, tk.ASSISTANT)
        for t in self._textos(self._fase(max_think, temperature, estruturais + (EOS_ID,), (tk.ANSWER,))):
            yield ('think', t)
        self._alimentar([tk.ANSWER])  # emitido pelo modelo OU forçado
        for t in self._textos(self._fase(max_answer, temperature, estruturais + (tk.ANSWER,), (EOS_ID,))):
            yield ('answer', t)


def _chave_voto(resposta: str) -> str:
    """Normaliza a resposta pra votação (ignora caixa, pontuação e espaços a mais)."""
    return re.sub(r'\W+', ' ', resposta.lower()).strip()


def raciocinar_com_votos(ger: Gerador, pergunta: str, cfg: dict, max_answer: int):
    """Roda `votos` tentativas silenciosas e devolve (pensamento, resposta, votos_vencedores, total)."""
    tentativas = []
    for _ in range(cfg['votos']):
        pensou, resp = [], []
        for fase, t in ger.raciocinar(pergunta, cfg['max_think'], max_answer, cfg['temp']):
            (pensou if fase == 'think' else resp).append(t)
        tentativas.append((''.join(pensou).strip(), ''.join(resp).strip()))
        yield ('progresso', len(tentativas))
    contagem = Counter(_chave_voto(r) for _, r in tentativas if r)
    if not contagem:
        yield ('final', (tentativas[0][0], tentativas[0][1], 0, len(tentativas)))
        return
    vencedora, n = contagem.most_common(1)[0]
    for p, r in tentativas:  # primeira tentativa com a resposta vencedora
        if _chave_voto(r) == vencedora:
            yield ('final', (p, r, n, len(tentativas)))
            return


# ═══════════════════════════════════════════════════════════════════════
# LOOP DE CHAT
# ═══════════════════════════════════════════════════════════════════════

def _ajuda() -> str:
    niveis = ' · '.join(NIVEIS)
    return (
        "[dim]Comandos: /sair · /reset · /raciocinio <" + niveis + "> (ou /r) · "
        "/pensar mostra/esconde o raciocínio · /temp 0.9 · /max 300[/dim]"
    )


def _out(texto: str, estilo: str = None):
    console.print(texto, end='', style=estilo, markup=False, highlight=False, soft_wrap=True)


def _responder(ger: Gerador, msg: str, historico: list, nivel: str, mostrar_pensamento: bool,
               max_tokens: int, temperature: float):
    """Gera e imprime a resposta. Devolve o texto da resposta final."""
    cfg = NIVEIS[nivel]

    # off: conversa direta com histórico
    if nivel == 'off':
        console.print('[bold magenta]IA:[/bold magenta] ', end='')
        partes = []
        for t in ger.chat(msg, historico, max_tokens, temperature):
            _out(t)
            partes.append(t)
        console.print()
        return ''.join(partes).strip()

    # alto: várias tentativas + votação (silencioso, com spinner)
    if cfg['votos'] > 1:
        resultado = None
        with console.status(f'[bold cyan]🧠 Pensando (0/{cfg["votos"]})...[/bold cyan]', spinner='dots') as st:
            for tipo, val in raciocinar_com_votos(ger, msg, cfg, max_tokens):
                if tipo == 'progresso':
                    st.update(f'[bold cyan]🧠 Pensando ({val}/{cfg["votos"]})...[/bold cyan]')
                else:
                    resultado = val
        pensou, resposta, n, total = resultado
        if mostrar_pensamento and pensou:
            console.print(f'[dim]💭 {pensou}[/dim]', highlight=False, markup=False)
        console.print('[bold magenta]IA:[/bold magenta] ', end='')
        _out(resposta)
        console.print(f'  [dim]({n}/{total} tentativas concordaram)[/dim]')
        return resposta

    # baixo / medio: uma tentativa, em streaming (pensamento em cinza, depois a resposta)
    fase_atual = None
    resposta = []
    for fase, t in ger.raciocinar(msg, cfg['max_think'], max_tokens, cfg['temp']):
        if fase != fase_atual:
            if fase_atual == 'think':
                console.print()
            if fase == 'think' and mostrar_pensamento:
                _out('💭 ', 'dim')
            elif fase == 'answer':
                console.print('[bold magenta]IA:[/bold magenta] ', end='')
            fase_atual = fase
        if fase == 'think':
            if mostrar_pensamento:
                _out(t, 'dim')
        else:
            _out(t)
            resposta.append(t)
    if fase_atual is None or fase_atual == 'think':  # (nunca deveria acontecer: a resposta é sempre iniciada)
        console.print('[bold magenta]IA:[/bold magenta] ', end='')
    console.print()
    return ''.join(resposta).strip()


def main():
    ap = argparse.ArgumentParser(description='Chat de terminal com um checkpoint do ia_server.py')
    ap.add_argument('--ckpt', default=None, help='Caminho do checkpoint .pt (padrão: checkpoint_last.pt)')
    ap.add_argument('--max-tokens', type=int, default=200, help='Máximo de tokens da resposta (padrão: 200)')
    ap.add_argument('--temperature', type=float, default=0.8, help='Temperatura do modo chat/off (padrão: 0.8)')
    ap.add_argument('--raciocinio', default='medio', help='Nível inicial: off | baixo | medio | alto (padrão: medio)')
    ap.add_argument('--esconder-pensamento', action='store_true', help='Não mostra o raciocínio (só a resposta)')
    ap.add_argument('--cpu', action='store_true', help='Força CPU mesmo com GPU disponível')
    args = ap.parse_args()

    nivel = _normalizar_nivel(args.raciocinio)
    if nivel is None:
        console.print(f'[yellow]Nível "{args.raciocinio}" inválido, usando medio.[/yellow]')
        nivel = 'medio'

    ckpt_path = args.ckpt or _default_checkpoint_path()
    model, tk, device = carregar_modelo(ckpt_path, force_cpu=args.cpu)
    ger = Gerador(model, tk, device)

    max_tokens = args.max_tokens
    temperature = args.temperature
    mostrar_pensamento = not args.esconder_pensamento
    historico = []  # [(usuario, ia), ...] — usado no modo off; os outros níveis também alimentam

    console.print(_ajuda())
    console.print(f'[dim]Raciocínio: [bold]{nivel}[/bold] — {NIVEIS[nivel]["desc"]}[/dim]')
    console.print()

    while True:
        try:
            msg = console.input(f'[bold cyan]Você[/bold cyan] [dim]({nivel})[/dim][bold cyan]:[/bold cyan] ').strip()
        except (KeyboardInterrupt, EOFError):
            console.print('\n[bold blue]Até mais! 👋[/bold blue]')
            break

        if not msg:
            continue

        if msg in ('/sair', '/exit', '/quit'):
            console.print('[bold blue]Até mais! 👋[/bold blue]')
            break
        if msg == '/reset':
            historico = []
            console.print('[dim]Histórico limpo.[/dim]')
            continue
        if msg == '/pensar':
            mostrar_pensamento = not mostrar_pensamento
            console.print(f'[dim]Raciocínio {"visível" if mostrar_pensamento else "escondido"}.[/dim]')
            continue
        if msg.startswith(('/raciocinio', '/raciocínio', '/r ')) or msg == '/r':
            partes = msg.split()
            if len(partes) == 2:
                novo = _normalizar_nivel(partes[1])
                if novo:
                    nivel = novo
                    console.print(f'[dim]Raciocínio: [bold]{nivel}[/bold] — {NIVEIS[nivel]["desc"]}[/dim]')
                    continue
            console.print('[yellow]Uso: /raciocinio off | baixo | medio | alto[/yellow]')
            for nome, c in NIVEIS.items():
                console.print(f'  [bold]{nome}[/bold] — {c["desc"]}')
            continue
        if msg.startswith('/temp'):
            partes = msg.split()
            if len(partes) == 2:
                try:
                    temperature = float(partes[1])
                    console.print(f'[dim]Temperatura (modo off) ajustada para {temperature}[/dim]')
                except ValueError:
                    console.print('[yellow]Uso: /temp 0.9[/yellow]')
            continue
        if msg.startswith('/max'):
            partes = msg.split()
            if len(partes) == 2:
                try:
                    max_tokens = int(partes[1])
                    console.print(f'[dim]Máximo de tokens ajustado para {max_tokens}[/dim]')
                except ValueError:
                    console.print('[yellow]Uso: /max 300[/yellow]')
            continue

        try:
            resposta = _responder(ger, msg, historico, nivel, mostrar_pensamento, max_tokens, temperature)
        except KeyboardInterrupt:
            console.print('\n[dim](geração interrompida)[/dim]')
            continue

        if resposta:
            historico.append((msg, resposta))
            historico = historico[-20:]  # teto de turnos; o chat() ainda corta por tokens


if __name__ == '__main__':
    main()
