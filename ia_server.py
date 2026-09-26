import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess, sys, time, random, math, os, threading, queue, select, csv
import io, gzip, base64, inspect, uuid
from collections import deque
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.console import Console, Group
from rich import box
try:
    import requests
except ImportError:
    requests = None
try:
    from flask import Flask, request, jsonify, send_file
except ImportError:
    Flask = None
console = Console()
_server_lock = threading.Lock()
_cmd_queue: 'queue.Queue[str]' = queue.Queue()

def _stdin_watcher():
    while True:
        try:
            r, _, _ = select.select([sys.stdin], [], [], 1.0)
            if r:
                line = sys.stdin.readline().strip()
                if line:
                    _cmd_queue.put(line)
        except Exception:
            break
_stdin_thread = threading.Thread(target=_stdin_watcher, daemon=True)
_stdin_thread.start()
SEED = 42
CONTEXT_LEN = 64  # era 64 — muito curto para aprender dependências de frase/parágrafo
PRETRAIN_STEPS = 3000
PRETRAIN_LR = 0.0006  # era 0.003 — alto demais pra ~14M params, arriscava instabilidade
PRETRAIN_LR_CONT = 0.0003  # era 0.0008
REINFORCE_LR = 0.0005
SUPERVISED_LR = 0.0005  # era 0.001
MAX_CODE_LEN = 64
CODE_TIMEOUT = 3
MUTATION_EVERY = 300
MUTATION_TRIALS = 3
MUTATION_STEPS = 80
REWARD_WINDOW = 30
TEMP_START = 1.1
TEMP_MIN = 0.6
TEMP_DECAY = 0.9997
ENTROPY_COEF = 0.03
ENTROPY_STUCK_THRESHOLD = 2.4
ENTROPY_STUCK_REWARD = 0.25
ENTROPY_STUCK_PATIENCE = 30
ENTROPY_STUCK_PT_STEPS = 3000
_KAGGLE_WORKING = '/kaggle/working'
if os.path.isdir(_KAGGLE_WORKING):
    CHECKPOINT_DIR = os.path.join(_KAGGLE_WORKING, 'checkpoints')
else:
    CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_EVERY = 50
CHECKPOINT_LAST = 'checkpoint_last.pt'
CHECKPOINT_BEST = 'checkpoint_best.pt'
PREMIUM_DIR = os.path.join(CHECKPOINT_DIR, 'premium')
PREMIUM_MIN_REWARD = 0.8
PREMIUM_MIN_FACTOR = 0.6
PREMIUM_MAX_SKILLS = 100
INIT_CONFIG = dict(embed_dim=256, n_heads=8, n_layers=6, dropout=0.1, use_moe=True, n_experts=4, top_k=1)
# Tier 1 (~14M params totais, ~poucos M ativos por token já que top_k=1).
# Tier 2 (~43M): embed_dim=384, n_heads=8,  n_layers=8,  n_experts=4, top_k=1, CONTEXT_LEN=320
# Tier 3 (~95M): embed_dim=512, n_heads=8,  n_layers=10, n_experts=4, top_k=1, CONTEXT_LEN=384
LEARN_SUP_STEPS = 400
LEARN_RL_TRIES = 300
LEARN_REPS_NEEDED = 40
LEARN_TEMP_START = 1.0
EXPAND_EMBED_DELTA = 16
MAX_CODE_LEN_LEARN = 120
torch.manual_seed(SEED)
random.seed(SEED)
CSV_MAX_CHARS_POR_ARQUIVO = 2000000
CSV_SAMPLE_LINHAS = 200
CSV_MIN_LEN_MEDIO_TEXTO = 15

# LIMITE MÁXIMO DE ARQUIVOS E CARACTERES (NOVO - RAILWAY FIX)
CORPUS_MAX_FILES = 50  # Máximo 50 arquivos
CORPUS_MAX_CHARS_TOTAL = 5000000  # 5MB total
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
_CATEGORIAS_CORPUS = {'code': ('.py',), 'general': ('.txt', '.csv')}
_corpus_por_categoria = carregar_corpus_categorizado([diretorio_raiz] + EXTRA_CORPUS_DIRS, _CATEGORIAS_CORPUS)
CORPUS_CODE = _corpus_por_categoria['code']
CORPUS_GENERAL = _corpus_por_categoria['general']
CORPUS = CORPUS_CODE + '\n' + CORPUS_GENERAL
EOS = '\x00'
chars = sorted(set(CORPUS + EOS))
VOCAB = len(chars)
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}
encode = lambda s: [c2i[c] for c in s if c in c2i]
decode = lambda l: ''.join((i2c.get(i, '?') for i in l))
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
        loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1)) if targets is not None else None
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
FED_AGG_THRESHOLD = 3               # quantos deltas válidos esperar antes de agregar
FED_JOB_TTL_SECONDS = 1800          # jobs emitidos e nunca respondidos expiram
FED_MAX_PENDING_JOBS = 500          # limite de jobs "em aberto" guardados em memória

_fed_lock = threading.Lock()
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
        'vocab_chars': chars,
        'context_len': CONTEXT_LEN,
        'weights': weights_b64,
        'max_steps': FED_MAX_STEPS_PER_JOB,
    }
    if not own_content:
        if len(CORPUS) > FED_JOB_CONTENT_CHARS:
            start = random.randint(0, len(CORPUS) - FED_JOB_CONTENT_CHARS)
            payload['content'] = CORPUS[start:start + FED_JOB_CONTENT_CHARS]
        else:
            payload['content'] = CORPUS
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
    """Faz a média ponderada (por n_steps) dos deltas pendentes e aplica no
    modelo mestre. Chamado dentro de _server_lock em quem chama de fora, ou
    aqui mesmo se for chamado isoladamente (idempotente com _fed_lock)."""
    with _fed_lock:
        deltas = _fed_state['pending_deltas']
        _fed_state['pending_deltas'] = []
        if not deltas:
            return _fed_state['version']
        total_steps = sum(n for _, n, _ in deltas)
        with _server_lock:
            sd = model.state_dict()
            for key in sd:
                if not torch.is_floating_point(sd[key]):
                    continue  # não mistura buffers inteiros (ex: máscara causal) na média
                acc = torch.zeros_like(sd[key])
                for delta, n_steps, _worker in deltas:
                    if key in delta:
                        acc += delta[key].to(sd[key].dtype) * (n_steps / total_steps)
                sd[key] = sd[key] + acc
            model.load_state_dict(sd)
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
    payload = {'model_cfg': model._cfg, 'model_state': model.state_dict(), 'vocab_chars': chars, 'vocab_size': VOCAB, 'rl_opt_state': rl_opt.state_dict(), 'sup_opt_state': sup_opt.state_dict(), 'pretrain_opt_state': pretrain_opt.state_dict() if pretrain_opt is not None else None, 'gen': state['gen'], 'temp': state['temp'], 'best_reward': state['best_reward'], 'best_code': state['best_code'], 'best_gen': state['best_gen'], 'config': state['config'], 'mutation_log': state['mutation_log'], 'reward_hist': list(state['reward_hist']), 'temp_resets': state['temp_resets'], 'stdout_memory': list(state.get('stdout_memory', [])), 'stdout_norm_mem': list(state.get('stdout_norm_memory', []))}
    torch.save(payload, path)

def load_checkpoint(filename: str=CHECKPOINT_LAST):
    path = _ckpt_path(filename)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt

def _premium_slug(stdout: str, code: str) -> str:
    import re as _re2
    slug = stdout.strip()[:40]
    slug = _re2.sub('[^a-zA-Z0-9 ]', '', slug)
    slug = slug.strip().lower()
    slug = _re2.sub('\\s+', '_', slug)
    slug = slug[:30] or 'skill'
    return slug

def save_premium_checkpoint(model: 'TinyAI', rl_opt, sup_opt, pretrain_opt, state: dict, code: str, stdout: str, r_val: float, gen: int, *, premium_stdout_set: set) -> bool:
    import re as _re2
    if r_val < PREMIUM_MIN_REWARD:
        return False
    factor = _code_complexity_factor(code)
    if factor < PREMIUM_MIN_FACTOR:
        return False

    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = _re2.sub('\\s+', ' ', s)
        s = _re2.sub('[^a-z0-9 ]', '', s)
        s = _re2.sub('(.)\\1{2,}', '\\1', s)
        return s.strip()
    norm_out = _norm(stdout) if stdout else ''
    if norm_out in premium_stdout_set:
        return False
    os.makedirs(PREMIUM_DIR, exist_ok=True)
    existing = [f for f in os.listdir(PREMIUM_DIR) if f.endswith('.pt')]
    if len(existing) >= PREMIUM_MAX_SKILLS:
        return False
    premium_stdout_set.add(norm_out)
    slug = _premium_slug(stdout, code)
    seq = len(existing) + 1
    filename = f'skill_{seq:03d}_r{r_val:.2f}_gen{gen}_{slug}.pt'
    path = os.path.join(PREMIUM_DIR, filename)
    payload = {'model_cfg': model._cfg, 'model_state': model.state_dict(), 'vocab_chars': chars, 'vocab_size': VOCAB, 'rl_opt_state': rl_opt.state_dict(), 'sup_opt_state': sup_opt.state_dict(), 'pretrain_opt_state': pretrain_opt.state_dict() if pretrain_opt else None, 'skill_code': code, 'skill_stdout': stdout, 'skill_reward': r_val, 'skill_gen': gen, 'skill_complexity': factor, 'gen': state['gen'], 'temp': state['temp'], 'best_reward': state['best_reward'], 'best_code': state['best_code'], 'best_gen': state['best_gen'], 'config': state['config'], 'mutation_log': state['mutation_log'], 'reward_hist': list(state['reward_hist']), 'temp_resets': state['temp_resets'], 'stdout_memory': list(state.get('stdout_memory', [])), 'stdout_norm_mem': list(state.get('stdout_norm_memory', []))}
    torch.save(payload, path)
    return True

def _init_premium_stdout_set() -> set:
    import re as _re2

    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = _re2.sub('\\s+', ' ', s)
        s = _re2.sub('[^a-z0-9 ]', '', s)
        s = _re2.sub('(.)\\1{2,}', '\\1', s)
        return s.strip()
    seen = set()
    if not os.path.isdir(PREMIUM_DIR):
        return seen
    for fname in os.listdir(PREMIUM_DIR):
        if not fname.endswith('.pt'):
            continue
        try:
            ckpt = torch.load(os.path.join(PREMIUM_DIR, fname), map_location='cpu', weights_only=False)
            out = ckpt.get('skill_stdout', '')
            if out:
                seen.add(_norm(out))
        except Exception:
            pass
    return seen

def _transplant_state_dict(saved_sd: dict, model: 'TinyAI') -> None:
    dst_sd = model.state_dict()
    for key in dst_sd:
        if key not in saved_sd:
            continue
        s, d = (saved_sd[key], dst_sd[key])
        if s.shape == d.shape:
            dst_sd[key] = s.clone()
        else:
            slices = tuple((slice(0, min(a, b)) for a, b in zip(s.shape, d.shape)))
            dst_sd[key][slices] = s[slices].clone()
    model.load_state_dict(dst_sd)

def restore_from_checkpoint(ckpt: dict, state: dict):
    model = TinyAI(ckpt['model_cfg']).to(device)
    ckpt_vocab = ckpt['model_state']['tok_emb.weight'].shape[0]
    vocab_mismatch = ckpt_vocab != VOCAB
    if vocab_mismatch:
        delta = VOCAB - ckpt_vocab
        console.print(f'[bold yellow]⚠️  Vocab mudou: checkpoint={ckpt_vocab} chars  atual={VOCAB} chars  ({delta:+d})[/bold yellow]\n   → Transplantando pesos compatíveis; {abs(delta)} token(s) {('novo(s)' if delta > 0 else 'removido(s)')} ficam com init aleatório.\n   → Pré-treino de recuperação será executado automaticamente.')
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

def generate_rl(model, temperature=0.9, on_char=None):
    model.train()
    log_probs = []
    entropies = []
    chars_so_far = []
    tok = torch.tensor([[c2i.get('\n', 0)]], dtype=torch.long, device=device)
    past_kv = None
    for _ in range(MAX_CODE_LEN):
        logits, _, past_kv = model(tok, past_kv=past_kv, use_cache=True)
        logits = logits[:, -1, :] / max(temperature, 0.1)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        sampled = dist.sample()
        lp = dist.log_prob(sampled)
        ent = -(probs * (probs + 1e-08).log()).sum()
        log_probs.append(lp)
        entropies.append(ent)
        ch = i2c.get(sampled.item(), '?')
        chars_so_far.append(ch)
        if on_char:
            on_char(''.join(chars_so_far))
        full = ''.join(chars_so_far)
        if EOS in full or full.count('\n') >= 2:
            break
        tok = sampled.unsqueeze(0)
    code = ''.join(chars_so_far).strip().replace(EOS, '')
    lp_t = torch.stack(log_probs) if log_probs else None
    ent_t = torch.stack(entropies) if entropies else None
    return (code, lp_t, ent_t)

def safe_exec(code: str):
    if not code.strip():
        return (-3, '', 'código vazio')
    try:
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=CODE_TIMEOUT)
        return (r.returncode, r.stdout.strip()[:200], r.stderr.strip()[:200])
    except subprocess.TimeoutExpired:
        return (-1, '', 'TIMEOUT')
    except Exception as e:
        return (-2, '', str(e)[:100])
import re as _re

def _is_only_comments(code: str) -> bool:
    for line in code.split('\n'):
        stripped = line.strip()
        if stripped and (not stripped.startswith('#')):
            return False
    return True
_COMPLEXITY_KW = ['for ', 'while ', 'if ', 'elif ', 'else:', 'def ', 'class ', 'import ', 'from ', 'return ', 'yield ', 'range(', 'len(', 'sum(', 'list(', 'dict(', 'set(', 'map(', 'lambda ', 'try:', 'except ', 'with ', 'open(', 'math.', 'random.']
_TRIVIAL_PRINT_RE = _re.compile('^print\\s*\\(\\s*(?:["\\\'][^"\\\']*["\\\']|\\d+)\\s*\\)$')
_SIMPLE_ASSIGN_RE = _re.compile('^[a-zA-Z_][a-zA-Z0-9_]*\\s*=\\s*.+$')
_REAL_FLOW_KW = ['for ', 'while ', 'if ', 'elif ', 'else:', 'def ', 'class ', 'return ', 'yield ', 'range(', 'len(', 'sum(', 'list(', 'dict(', 'set(', 'map(', 'try:', 'except ', 'with ', 'open(', 'math.', 'random.']
_WEAK_KW = ['import ', 'from ', 'lambda ']

def _code_complexity_factor(code: str) -> float:
    lines = [l.strip() for l in code.split('\n') if l.strip() and (not l.strip().startswith('#'))]
    if not lines:
        return 0.02
    has_real_flow = any((kw in code for kw in _REAL_FLOW_KW))
    has_weak_kw = any((kw in code for kw in _WEAK_KW))
    print_lines = [l for l in lines if l.startswith('print')]
    non_print_lines = [l for l in lines if not l.startswith('print')]
    if has_real_flow:
        return 1.4
    if has_weak_kw and (not print_lines):
        return 0.8
    if print_lines and (not non_print_lines):
        return 0.15
    if not print_lines and all((_SIMPLE_ASSIGN_RE.match(l) for l in lines)):
        return 0.1
    if print_lines and non_print_lines:
        return 0.5
    return 0.8

def reward(rc, stdout, stderr, code: str='') -> float:
    if rc in (-1, -3):
        return 0.0
    if code and _is_only_comments(code):
        return 0.01
    if rc != 0:
        factor = _code_complexity_factor(code) if code else 1.0
        base = 0.05 if 'SyntaxError' in stderr else 0.2
        return round(min(base * factor, base), 3)
    factor = _code_complexity_factor(code) if code else 1.0
    if not stdout:
        return round(min(0.12 * factor, 0.2), 3)
    sl = stdout.lower().strip()
    canonical = {'hi', 'hello', 'hey', 'hi there', 'hello world', 'hey there'}
    if sl in canonical:
        return 2.0
    if any((w in sl for w in ('hi', 'hello', 'hey'))):
        return round(1.0 * max(factor, 0.5), 3)
    return round(min(0.7 * factor, 0.98), 3)

def get_batch(bs=1):  # era 4 — muito pequeno pra ~14M params, gradiente ruidoso demais
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
        if status_callback:
            status_callback(step, steps, loss.item())
    return model

def _get_batch_de_texto(texto_encoded: torch.Tensor, bs=16):  # era 4
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

def reinforce_update(model, optimizer, log_probs, entropies, r_val, baseline):
    if log_probs is None or entropies is None or len(log_probs) == 0:
        return (0.0, 0.0)
    advantage = r_val - baseline
    policy_loss = -log_probs.sum() * advantage
    entropy_bonus = -ENTROPY_COEF * entropies.mean()
    loss = policy_loss + entropy_bonus
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return (loss.item(), entropies.mean().item())

def supervised_on_success(model, optimizer, code, steps=20):
    toks = encode(code + '\n')
    if len(toks) < 4:
        return
    t = torch.tensor(toks, dtype=torch.long, device=device)
    for _ in range(steps):
        x = t[:-1].unsqueeze(0)
        y = t[1:].unsqueeze(0)
        if x.shape[1] < 1:
            break
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

def transplant_weights(src: TinyAI, dst: TinyAI) -> None:
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for key in dst_sd:
        if key not in src_sd:
            continue
        s, d = (src_sd[key], dst_sd[key])
        if s.shape == d.shape:
            dst_sd[key] = s.clone()
        else:
            slices = tuple((slice(0, min(a, b)) for a, b in zip(s.shape, d.shape)))
            dst_sd[key][slices] = s[slices].clone()
    dst.load_state_dict(dst_sd)

def mutate_config(cfg: dict) -> dict:
    new = cfg.copy()
    kind = random.choice(['embed', 'layers', 'heads', 'dropout', 'experts'])
    if kind == 'embed':
        new['embed_dim'] = max(32, cfg['embed_dim'] + random.choice([-16, 16, 32]))
        new['embed_dim'] = new['embed_dim'] // new['n_heads'] * new['n_heads']
    elif kind == 'layers':
        new['n_layers'] = max(1, min(8, cfg['n_layers'] + random.choice([-1, 1])))
    elif kind == 'heads':
        new['n_heads'] = random.choice([2, 4, 8])
        new['embed_dim'] = max(new['n_heads'] * 8, new['embed_dim'])
        new['embed_dim'] = new['embed_dim'] // new['n_heads'] * new['n_heads']
    elif kind == 'dropout':
        new['dropout'] = round(max(0.0, min(0.4, cfg['dropout'] + random.choice([-0.05, 0.05]))), 2)
    elif kind == 'experts' and cfg.get('use_moe'):
        choices = [e for e in [2, 4, 8] if e != cfg.get('n_experts', 4)]
        new['n_experts'] = random.choice(choices)
        new['top_k'] = min(new['n_experts'] - 1, cfg.get('top_k', 2))
    return new

def _eval_model_loss(model, n_batches=20):
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(16)
            if x is None:
                break
            _, l, _ = model(x, y)
            losses.append(l.item())
    return sum(losses) / max(1, len(losses))

def eval_config(cfg: dict):
    m = TinyAI(cfg).to(device)
    pretrain(m, MUTATION_STEPS, PRETRAIN_LR)
    return (m, _eval_model_loss(m))

def eval_mutation_from_model(base_model: TinyAI, cfg: dict, steps: int=MUTATION_STEPS):
    m = TinyAI(cfg).to(device)
    transplant_weights(base_model, m)
    pretrain(m, steps, PRETRAIN_LR)
    return (m, _eval_model_loss(m))
REWARD_COLORS = {0.0: 'red', 0.05: 'red', 0.2: 'orange3', 0.5: 'yellow', 0.7: 'green', 1.0: 'bright_green', 2.0: 'bold magenta'}

def reward_color(r_val):
    for thresh in sorted(REWARD_COLORS.keys(), reverse=True):
        if r_val >= thresh:
            return REWARD_COLORS[thresh]
    return 'white'

def sparkline(values, width=28):
    BLOCKS = '▁▂▃▄▅▆▇█'
    if not values:
        return '─' * width
    mx = max(values) or 1
    return ''.join((BLOCKS[min(7, int(v / mx * 7))] for v in list(values)[-width:]))

def make_display(st):
    cfg = st['config']
    hist = st['reward_hist']
    avg_r = sum(list(hist)[-20:]) / max(1, min(20, len(hist)))
    phase_lbl = '[bold cyan]PRÉ-TREINO[/]' if st['phase'] == 'pretrain' else '[bold green]EVOLUINDO[/]'
    div = st.get('diversity', 1.0)
    div_color = 'bright_green' if div >= 0.5 else 'yellow' if div >= 0.3 else 'bold red blink'
    div_str = f'[{div_color}]{div:.0%}[/]'
    ent = st.get('last_entropy', 0.0)
    ent_color = 'bright_green' if ent > 2.0 else 'yellow' if ent > 1.0 else 'bold red'
    stuck = st.get('entropy_stuck_count', 0)
    stuck_str = f'  │  [bold red blink]🆘 EntStuck:{stuck}/{ENTROPY_STUCK_PATIENCE}[/]' if stuck > ENTROPY_STUCK_PATIENCE // 2 else ''
    ckpt_str = f'💾 ckpt@gen{st.get('last_ckpt_gen', '—')}' if st.get('last_ckpt_gen') else '💾 sem ckpt'
    header = Panel(f'{phase_lbl}  │  Gen: [bold]{st['gen']}[/]  │  Melhor: [bold {reward_color(st['best_reward'])}]{st['best_reward']:.2f}[/]  │  Avg(20): [bold]{avg_r:.2f}[/]  │  Div: {div_str}  │  Entropia: [{ent_color}]{ent:.2f}[/]  │  Temp: {st['temp']:.3f}  │  Resets: {st.get('temp_resets', 0)}  │  {ckpt_str}  │  [dim]⌨️  P [+steps]=pré-treino  E id arq [+steps]=treino expert[/dim]  │  {dev_str}{stuck_str}', style='bold blue', box=box.HEAVY)
    code_str = st['current_code'] or '[dim]aguardando...[/dim]'
    code_text = Text(code_str, style='bold green')
    if st['phase'] == 'generate':
        code_text.append('█', style='blink bright_green')
    code_panel = Panel(code_text, title='[bold green]🖊️  GERANDO CÓDIGO (ao vivo)[/]', border_style='green', box=box.ROUNDED)
    rc, stdout, stderr = st.get('last_exec', (None, '', ''))
    if rc is None:
        exec_body = '[dim]nenhuma execução ainda[/dim]'
    elif rc == 0:
        exec_body = f'[bold green]✅ returncode: 0[/]\nstdout: [bright_white]{repr(stdout[:60])}[/]\nstderr: [dim]—[/dim]'
    else:
        tag = {-1: 'TIMEOUT ⏱', -3: 'VAZIO'}.get(rc, f'ERRO (rc={rc})')
        exec_body = f'[bold red]❌ {tag}[/]\nstdout: [dim]{repr(stdout[:40])}[/]\nstderr: [yellow]{stderr[:80]}[/yellow]'
    last_r = st.get('last_reward', 0.0)
    exec_panel = Panel(exec_body + f'\n[bold]reward: [{reward_color(last_r)}]{last_r:.2f}[/][/bold]', title='[bold yellow]⚡ EXECUÇÃO[/]', border_style='yellow', box=box.ROUNDED)
    best_body = f'[bold cyan]{st['best_code'] or 'nenhum ainda...'}[/]\nreward: [bold {reward_color(st['best_reward'])}]{st['best_reward']:.2f}[/]  │  na geração {st['best_gen']}'
    best_panel = Panel(best_body, title='[bold cyan]🏆 MELHOR RESULTADO[/]', border_style='cyan', box=box.ROUNDED)
    spark = sparkline(list(hist))
    recent = list(hist)[-10:]
    dist_str = ' '.join((f'[{reward_color(r)}]{r:.1f}[/]' for r in recent))
    hist_panel = Panel(f'[bold]{spark}[/]\n{dist_str}', title='[bold magenta]📊 HISTÓRICO DE REWARDS[/]', border_style='magenta', box=box.ROUNDED)
    gens_to_mut = MUTATION_EVERY - st['gen'] % MUTATION_EVERY
    mut_log = '  │  '.join(st['mutation_log'][-3:]) if st['mutation_log'] else '—'
    moe_info = ''
    if cfg.get('use_moe'):
        moe_info = f'  │  [bold]MoE[/]: experts={cfg.get('n_experts', 4)} top-k={cfg.get('top_k', 2)}'
    arch_panel = Panel(f'embed={cfg['embed_dim']}  layers={cfg['n_layers']}  heads={cfg['n_heads']}  dropout={cfg['dropout']}{moe_info}  │  params={st['n_params']:,}  │  próxima mutação em: [bold]{gens_to_mut}[/] gens\nhist: {mut_log}', title='[bold blue]🧬 ARQUITETURA (MoE + auto-evolução)[/]', border_style='blue', box=box.ROUNDED)
    log_lines = '\n'.join(st['log'][-5:]) or '[dim]...[/dim]'
    log_panel = Panel(log_lines, title='[dim]📝 LOG[/dim]', border_style='dim', box=box.SIMPLE)
    return Group(header, Columns([code_panel, exec_panel], equal=True), Columns([best_panel, hist_panel], equal=True), arch_panel, log_panel)

def make_throttled_updater(live, state, min_interval: float=0.1):
    _last = [0.0]

    def update(force: bool=False):
        now = time.monotonic()
        if force or now - _last[0] >= min_interval:
            live.update(make_display(state))
            _last[0] = now
    return update

def novo_estado_inicial():
    return dict(phase='pretrain', gen=0, temp=TEMP_START, config=INIT_CONFIG.copy(), n_params=0, current_code='', last_exec=(None, '', ''), last_reward=0.0, best_code='', best_reward=0.0, best_gen=0, reward_hist=deque(maxlen=REWARD_WINDOW * 3), mutation_log=[], log=[], code_memory=set(), stdout_memory=set(), stdout_norm_memory=set(), recent_codes=deque(maxlen=30), diversity=0.0, last_entropy=0.0, temp_resets=0, last_ckpt_gen=None, entropy_stuck_count=0, premium_count=0)

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
    console.print("[dim]Rode 'python ia.py' sem argumentos pra voltar ao loop normal de auto-evolução com esse expert já treinado.[/dim]\n")

def main():
    console.print('\n[bold blue]══ 🧬 SELF-EVOLVING AI v4  (MoE + Checkpoints + Premium Skills) ══[/bold blue]')
    console.print(f'   {dev_str}')
    console.print(f'   vocab={VOCAB} chars  |  context={CONTEXT_LEN}  |  corpus={len(data_tensor)} tokens\n')
    state = novo_estado_inicial()
    premium_stdout_set = _init_premium_stdout_set()
    if premium_stdout_set:
        console.print(f"[bold magenta]🌟 Premium: {len(premium_stdout_set)} habilidade(s) já salva(s) em '{PREMIUM_DIR}'[/bold magenta]")

    def log(msg):
        state['log'].append(msg)
    ckpt = load_checkpoint(CHECKPOINT_LAST)
    if ckpt is not None:
        console.print('[bold yellow]♻️  Checkpoint encontrado! Retomando treino...[/bold yellow]')
        model, rl_opt, sup_opt, pretrain_opt, vocab_mismatch = restore_from_checkpoint(ckpt, state)
        state['last_ckpt_gen'] = state['gen']
        log(f'♻️  Retomado do checkpoint: gen={state['gen']}  best_reward={state['best_reward']:.2f}')
        skip_pretrain = not vocab_mismatch
        if vocab_mismatch:
            log(f'⚠️  Vocab mudou → pré-treino de recuperação agendado')
        else:
            avg_hist = sum(ckpt.get('reward_hist', [0])[-20:]) / max(1, min(20, len(ckpt.get('reward_hist', [0]))))
            if avg_hist < ENTROPY_STUCK_REWARD:
                skip_pretrain = False
                vocab_mismatch = True
                log(f'⚠️  Sessão anterior com avg_reward={avg_hist:.2f} < {ENTROPY_STUCK_REWARD} → pré-treino de re-ancoragem agendado')
    else:
        console.print('[bold green]🆕 Nenhum checkpoint encontrado. Começando do zero.[/bold green]')
        model = TinyAI(INIT_CONFIG).to(device)
        state['n_params'] = model.n_params
        rl_opt = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
        sup_opt = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
        pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.0001)
        skip_pretrain = False
        vocab_mismatch = False
        log(f'Modelo MoE criado: {model.n_params:,} params  (experts={INIT_CONFIG['n_experts']} top-k={INIT_CONFIG['top_k']})')
    with Live(make_display(state), refresh_per_second=10, screen=False) as live:
        refresh = make_throttled_updater(live, state, min_interval=0.1)
        if not skip_pretrain:
            state['phase'] = 'pretrain'
            pt_steps = 2500 if vocab_mismatch else PRETRAIN_STEPS
            pt_label = 'recuperação' if vocab_mismatch else 'inicial'
            log(f'Iniciando pré-treino de {pt_label} ({pt_steps} steps)...')
            refresh(force=True)

            def pretrain_cb(step, total, loss):
                state['current_code'] = f'[pré-treino {pt_label}] step {step}/{total}  loss={loss:.4f}'
                refresh()
                if step > 0 and step % 50 == 0:
                    save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
            pretrain(model, pt_steps, PRETRAIN_LR, pretrain_cb, opt=pretrain_opt)
            log(f'✅ Pré-treino de {pt_label} concluído. Iniciando auto-evolução...')
        state['phase'] = 'evolve'
        state['current_code'] = ''
        refresh(force=True)
        time.sleep(0.3)
        try:
            while True:
                state['gen'] += 1
                gen = state['gen']
                state['phase'] = 'generate'

                def on_char(s):
                    state['current_code'] = s
                    refresh()
                code, log_probs, entropies = generate_rl(model, temperature=state['temp'], on_char=on_char)
                state['current_code'] = code
                state['phase'] = 'exec'
                refresh(force=True)
                rc, stdout, stderr = safe_exec(code)
                r_val = reward(rc, stdout, stderr, code)
                state['last_exec'] = (rc, stdout, stderr)
                state['last_reward'] = r_val
                state['reward_hist'].append(r_val)
                code_hash = hash(code)
                is_duplicate = code_hash in state['code_memory']
                state['code_memory'].add(code_hash)
                state['recent_codes'].append(code)
                unique_recent = len(set(state['recent_codes']))
                state['diversity'] = unique_recent / max(1, len(state['recent_codes']))

                def _normalize_stdout(s: str) -> str:
                    s = s.lower().strip()
                    s = _re.sub('\\s+', ' ', s)
                    s = _re.sub('[^a-z0-9 ]', '', s)
                    s = _re.sub('(.)\\1{2,}', '\\1', s)
                    for seg_len in range(len(s) // 2, 1, -1):
                        seg = s[:seg_len]
                        if s == seg * (len(s) // seg_len) and len(s) % seg_len == 0:
                            s = seg
                            break
                    return s.strip()
                stdout_key = stdout.strip() if stdout else ''
                stdout_norm_key = _normalize_stdout(stdout_key) if stdout_key else ''
                stdout_is_new = stdout_key == '' or (stdout_key not in state['stdout_memory'] and stdout_norm_key not in state['stdout_norm_memory'])
                if stdout_key and rc == 0:
                    state['stdout_memory'].add(stdout_key)
                    if stdout_norm_key:
                        state['stdout_norm_memory'].add(stdout_norm_key)
                if is_duplicate:
                    rl_reward = 0.0
                    log(f'⚠️  Gen {gen}: duplicata exata — update ignorado')
                elif not stdout_is_new and r_val >= 0.5:
                    rl_reward = 0.0
                    log(f'⚠️  Gen {gen}: stdout repetido ({repr(stdout_key[:25])}) — sem reward RL')
                else:
                    rl_reward = r_val
                if r_val >= state['best_reward'] or not state['best_code']:
                    state['best_reward'] = r_val
                    state['best_code'] = code
                    state['best_gen'] = gen
                    log(f'🏆 Gen {gen}: novo recorde! reward={r_val:.2f}  →  {repr(code[:50])}')
                    save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_BEST, pretrain_opt=pretrain_opt)
                    log(f'💾 Best checkpoint salvo (gen {gen})')
                if rc == 0 and stdout and stdout_is_new:
                    saved = save_premium_checkpoint(model, rl_opt, sup_opt, pretrain_opt, state, code, stdout, r_val, gen, premium_stdout_set=premium_stdout_set)
                    if saved:
                        state['premium_count'] += 1
                        log(f'🌟 Gen {gen}: habilidade premium salva! reward={r_val:.2f}  stdout={repr(stdout[:30])}  total={state['premium_count']}')
                baseline = sum(list(state['reward_hist'])[-20:]) / max(1, min(20, len(state['reward_hist'])))
                if not is_duplicate:
                    _, ent_val = reinforce_update(model, rl_opt, log_probs, entropies, rl_reward, baseline)
                    if entropies is not None:
                        state['last_entropy'] = ent_val
                elif entropies is not None:
                    state['last_entropy'] = entropies.mean().item()
                if rc == 0 and stdout and (r_val >= 0.5) and (not is_duplicate) and stdout_is_new:
                    supervised_on_success(model, sup_opt, code, steps=15)
                    if r_val >= 1.0:
                        log(f'✨ Gen {gen}: funcionou! reward={r_val:.2f}  stdout={repr(stdout[:30])}')
                state['temp'] = max(TEMP_MIN, state['temp'] * TEMP_DECAY)
                if len(state['recent_codes']) >= 15 and state['diversity'] < 0.3 and (gen % 10 == 0):
                    old_temp = state['temp']
                    state['temp'] = min(TEMP_START, state['temp'] * 3.0)
                    state['temp_resets'] += 1
                    log(f'🔄 Gen {gen}: diversidade={state['diversity']:.0%} → temp reset {old_temp:.3f}→{state['temp']:.3f}')
                cur_avg = sum(list(state['reward_hist'])[-20:]) / max(1, min(20, len(state['reward_hist'])))
                if state['last_entropy'] >= ENTROPY_STUCK_THRESHOLD and cur_avg < ENTROPY_STUCK_REWARD:
                    state['entropy_stuck_count'] += 1
                else:
                    state['entropy_stuck_count'] = 0
                if state['entropy_stuck_count'] >= ENTROPY_STUCK_PATIENCE:
                    state['entropy_stuck_count'] = 0
                    log(f'🆘 Gen {gen}: entropia={state['last_entropy']:.2f} travada  avg={cur_avg:.2f} por {ENTROPY_STUCK_PATIENCE} gens → pré-treino automático de re-ancoragem ({ENTROPY_STUCK_PT_STEPS} steps)!')
                    state['phase'] = 'pretrain'
                    state['current_code'] = f'[re-ancoragem automática] 0/{ENTROPY_STUCK_PT_STEPS}'
                    refresh(force=True)

                    def _auto_pt_cb(step, total, loss):
                        state['current_code'] = f'[re-ancoragem automática] step {step}/{total}  loss={loss:.4f}'
                        refresh()
                        if step > 0 and step % 50 == 0:
                            save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
                    pretrain(model, ENTROPY_STUCK_PT_STEPS, PRETRAIN_LR_CONT, _auto_pt_cb, opt=pretrain_opt)
                    state['temp'] = TEMP_START
                    state['phase'] = 'evolve'
                    state['current_code'] = ''
                    log(f'✅ Re-ancoragem concluída. Temp resetada para {TEMP_START}')
                    refresh(force=True)
                if gen % CHECKPOINT_EVERY == 0:
                    save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
                    state['last_ckpt_gen'] = gen
                    log(f'💾 Checkpoint salvo (gen {gen})')
                if gen > 0 and gen % MUTATION_EVERY == 0:
                    state['phase'] = 'mutation'
                    log(f'🧬 Gen {gen}: avaliando mutações...')
                    refresh(force=True)
                    current_cfg = state['config']
                    ref_loss = _eval_model_loss(model)
                    best_mut_loss = ref_loss
                    best_mut_cfg = None
                    best_mut_m = None
                    for t in range(MUTATION_TRIALS):
                        mut_cfg = mutate_config(current_cfg)
                        state['current_code'] = f'[mutação {t + 1}/{MUTATION_TRIALS}] embed={mut_cfg['embed_dim']} layers={mut_cfg['n_layers']} heads={mut_cfg['n_heads']}' + (f' experts={mut_cfg.get('n_experts')}' if mut_cfg.get('use_moe') else '')
                        refresh()
                        try:
                            mut_m, mut_loss = eval_mutation_from_model(model, mut_cfg)
                        except Exception as e:
                            log(f'⚠️  Mutação {t + 1}/{MUTATION_TRIALS} falhou: {type(e).__name__}: {e}')
                            continue
                        if mut_loss < best_mut_loss:
                            best_mut_loss = mut_loss
                            best_mut_cfg = mut_cfg
                            best_mut_m = mut_m
                    if best_mut_cfg is not None:
                        model = best_mut_m
                        state['config'] = best_mut_cfg
                        state['n_params'] = model.n_params
                        rl_opt = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
                        sup_opt = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
                        pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.0001)
                        desc = f'embed={best_mut_cfg['embed_dim']} L={best_mut_cfg['n_layers']} H={best_mut_cfg['n_heads']}' + (f' E={best_mut_cfg.get('n_experts')}' if best_mut_cfg.get('use_moe') else '')
                        state['mutation_log'].append(f'✅ {desc} (gen {gen})')
                        log(f'🧬 Mutação aceita! {desc}  loss {ref_loss:.4f}→{best_mut_loss:.4f}')
                        save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
                        state['last_ckpt_gen'] = gen
                        log(f'💾 Checkpoint pós-mutação salvo')
                    else:
                        state['mutation_log'].append(f'❌ manteve (gen {gen})')
                        log(f'🧬 Sem melhora na mutação (ref_loss={ref_loss:.4f})')
                    state['phase'] = 'evolve'
                    state['current_code'] = ''
                try:
                    cmd = _cmd_queue.get_nowait()
                    parts_raw = cmd.strip().split()
                    parts = cmd.strip().upper().split()
                    if parts and parts[0] == 'P':
                        pt_steps = PRETRAIN_STEPS
                        if len(parts) > 1:
                            try:
                                pt_steps = int(parts[1])
                            except ValueError:
                                pass
                        log(f'⌨️  Pré-treino manual solicitado ({pt_steps} steps)...')
                        state['phase'] = 'pretrain'
                        state['current_code'] = f'[pré-treino manual] 0/{pt_steps}'
                        refresh()

                        def _manual_cb(step, total, loss):
                            state['current_code'] = f'[pré-treino manual] step {step}/{total}  loss={loss:.4f}'
                            refresh()
                            if step > 0 and step % 50 == 0:
                                save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
                        pretrain(model, pt_steps, PRETRAIN_LR_CONT, _manual_cb, opt=pretrain_opt)
                        state['phase'] = 'evolve'
                        state['current_code'] = ''
                        log(f'✅ Pré-treino manual concluído ({pt_steps} steps)')
                        refresh()
                    elif parts and parts[0] == 'E':
                        if len(parts_raw) < 3:
                            log('⌨️  Uso: E <expert_id> <arquivo> [steps]  (ex: E 0 ./dados/python.txt 800)')
                        else:
                            try:
                                exp_id = int(parts_raw[1])
                            except ValueError:
                                log(f'⌨️  expert_id inválido: {parts_raw[1]!r}')
                                exp_id = None
                            caminho_txt = parts_raw[2] if exp_id is not None else None
                            e_steps = LEARN_SUP_STEPS
                            if exp_id is not None and len(parts_raw) > 3:
                                try:
                                    e_steps = int(parts_raw[3])
                                except ValueError:
                                    pass
                            if exp_id is not None:
                                if not os.path.isfile(caminho_txt):
                                    log(f'⌨️  Arquivo não encontrado: {caminho_txt}')
                                else:
                                    with open(caminho_txt, 'r', encoding='utf-8', errors='ignore') as f:
                                        texto_expert = f.read()
                                    log(f"⌨️  Treinando expert {exp_id} com '{caminho_txt}' ({len(texto_expert)} chars, {e_steps} steps)...")
                                    state['phase'] = 'pretrain'
                                    state['current_code'] = f'[expert {exp_id}] 0/{e_steps}'
                                    refresh()

                                    def _expert_cb(step, total, loss, _eid=exp_id):
                                        state['current_code'] = f'[expert {_eid}] step {step}/{total}  loss={loss:.4f}'
                                        refresh()
                                    try:
                                        treinar_expert(model, exp_id, texto_expert, e_steps, SUPERVISED_LR, status_callback=_expert_cb)
                                        save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
                                        log(f'✅ Expert {exp_id} treinado e checkpoint salvo ({e_steps} steps)')
                                    except (ValueError, RuntimeError) as e:
                                        log(f'⚠️  Falha ao treinar expert {exp_id}: {e}')
                                    state['phase'] = 'evolve'
                                    state['current_code'] = ''
                                    refresh()
                except queue.Empty:
                    pass
                refresh(force=True)
        except KeyboardInterrupt:
            pass
    console.print('\n[yellow]💾 Salvando checkpoint final...[/yellow]')
    save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
    console.print('\n[bold blue]══ RESUMO FINAL ══[/bold blue]')
    console.print(f'   Gerações: {state['gen']}')
    console.print(f'   Melhor reward: {state['best_reward']:.2f}')
    console.print(f'   Melhor código (gen {state['best_gen']}):')
    console.print(f'   [bold cyan]{repr(state['best_code'])}[/bold cyan]')
    console.print(f'   Arquitetura final: {state['config']}')
    console.print(f'   Parâmetros: {state['n_params']:,}')
    console.print(f'   Checkpoints em: [bold]{os.path.abspath(CHECKPOINT_DIR)}[/bold]')
    console.print(f'   Habilidades premium: [bold magenta]{state['premium_count']}[/bold magenta] salvas em: [bold]{os.path.abspath(PREMIUM_DIR)}[/bold]\n')
    console.print('[bold green]🧬 A IA evoluiu. Até a próxima![/bold green]\n')
@torch.no_grad()
def generate_text(model, prompt: str, max_new: int=200, temperature: float=0.8) -> str:
    model.eval()
    tokens = encode(prompt)
    if not tokens:
        tokens = [c2i.get('\n', 0)]
    tok = torch.tensor([tokens[-CONTEXT_LEN:]], dtype=torch.long, device=device)
    out_chars = []
    logits, _, past_kv = model(tok, use_cache=True)
    next_logits = logits[:, -1, :] / max(temperature, 0.1)
    for _ in range(max_new):
        probs = F.softmax(next_logits, dim=-1)
        sampled = torch.distributions.Categorical(probs).sample()
        ch = i2c.get(sampled.item(), '?')
        if ch == EOS:
            break
        out_chars.append(ch)
        tok = sampled.view(1, 1)
        logits, _, past_kv = model(tok, past_kv=past_kv, use_cache=True)
        next_logits = logits[:, -1, :] / max(temperature, 0.1)
    model.train()
    return ''.join(out_chars)


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

def _wiki_random_page_text(lang: str='pt'):
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


def _wiki_training_loop(model, pretrain_opt, rl_opt, sup_opt, state, loss_target: float=0.7, max_steps_por_pagina: int=2000):
    console.print('[bold cyan]📖 Loop de treino com Wikipedia iniciado (--wiki)[/bold cyan]')
    while True:
        try:
            title, texto = _wiki_random_page_text()
        except Exception as e:
            console.print(f'[bold red]⚠️  Falha ao buscar página da Wikipedia: {e}[/bold red]')
            time.sleep(5)
            continue
        encoded = torch.tensor(encode(texto), dtype=torch.long)
        if len(encoded) <= CONTEXT_LEN + 1:
            continue
        console.print(f"[cyan]📖 Wiki: '{title}' ({len(texto)} chars)[/cyan]")
        step = 0
        last_loss = None
        while step < max_steps_por_pagina:
            x, y = _get_batch_de_texto(encoded)
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
            if step % 20 == 0:
                console.print(f'   step {step}  loss={last_loss:.4f}')
            if last_loss <= loss_target:
                break
        with _server_lock:
            state['gen'] = state.get('gen', 0) + 1
            save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
        motivo = 'atingiu loss alvo' if last_loss is not None and last_loss <= loss_target else 'limite de steps'
        console.print(f"[bold green]✅ '{title}' concluída ({motivo}) — loss final={last_loss:.4f}, {step} steps. Checkpoint salvo.[/bold green]")


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


def run_server(wiki: bool=False, host: str='0.0.0.0', port: int=5000, federated: bool=False):
    if Flask is None:
        console.print('[bold red]❌ Flask não instalado. Rode: pip install flask[/bold red]')
        return
    console.print('\n[bold blue]══ 🧬 SELF-EVOLVING AI — Servidor Flask ══[/bold blue]')
    state = novo_estado_inicial()
    model, rl_opt, sup_opt, pretrain_opt = _carregar_ou_criar_modelo(state)
    if wiki:
        t = threading.Thread(target=_wiki_training_loop, args=(model, pretrain_opt, rl_opt, sup_opt, state), daemon=True)
        t.start()
    app = _build_flask_app(model, rl_opt, sup_opt, pretrain_opt, state, federated=federated)
    rotas = 'GET /v1/model, POST /v1/chat, GET /v1/status'
    if federated:
        rotas += ', GET /v1/job, POST /v1/submit'
    console.print(f'[bold green]🚀 Servidor em http://{host}:{port}  ({rotas})[/bold green]\n')
    app.run(host=host, port=port, threaded=True)


if __name__ == '__main__':
    if '--serve' in sys.argv:
        _port = 5000
        if '--port' in sys.argv:
            try:
                _port = int(sys.argv[sys.argv.index('--port') + 1])
            except (ValueError, IndexError):
                pass
        run_server(wiki='--wiki' in sys.argv, port=_port, federated='--federated' in sys.argv)
    else:
        _cli = _parse_expert_cli(sys.argv)
        if _cli is not None:
            _expert_id, _categoria, _steps = _cli
            rodar_treino_expert_cli(_expert_id, _categoria, _steps)
        else:
            main()
