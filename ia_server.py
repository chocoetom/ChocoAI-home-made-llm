"""
ia_server.py — TinyAI: GPT pequeno e DENSO (sem MoE) com Chain-of-Thought (CoT).

Mudanças em relação à versão anterior
-------------------------------------
* MoE removido por completo (MoELayer, roteador, aux-loss, treino de experts, CLI de experts).
* Federado removido (rotas /v1/job e /v1/submit, FedAvg, envio de código do modelo).
* Modelo bem menor: preset 'small' ~16.6M params (antes 86M) — troque com MODEL_PRESET=tiny|small|base.
* Checkpoint ~200MB (antes ~1.1GB): sem máscara causal salva (usa F.scaled_dot_product_attention),
  sem otimizadores RL/SUP vazios, head amarrada ao embedding (weight tying), escrita atômica.
* /v1/model agora baixa só os pesos em fp16 (~33MB), sem estado do otimizador.
* CoT: 3 tokens especiais (<|q|>, <|think|>, <|answer|>), formato de treino, gerador sintético de
  raciocínio passo a passo, mistura CoT/texto nos batches e geração em duas fases (pensa -> responde).

Task tokens (modos)
-------------------
Cada modo começa com UM token de tarefa, que muda a distribuição condicional do modelo (sinal forte
mesmo com poucos exemplos — não depende de o modo ser frequente no corpus):
    <|chat|>  conversa:    <|chat|><|user|>Hello!\n<|assistant|>Hi! How can I help?\x00
    <|cot|>   raciocínio:  <|cot|><|q|>pergunta\n<|think|>passos\n<|answer|>resposta\x00
    (nenhum)  texto corrido (Wikipedia etc.)
Os dados de chat são gerados sinteticamente (en/pt, milhares de conversas: saudações, identidade, humor,
limites, matemática direta, capitais, fatos, multi-turno) e entram no batch com fração própria
(CHAT_BATCH_FRACTION), como o CoT. Pra usar conversas suas: um .txt com blocos
<|chat|><|user|>...\n<|assistant|>...\x00 (a conversa termina em \x00) é reconhecido como chat.

Formato CoT (o que o modelo aprende a produzir)
-----------------------------------------------
    <|cot|><|q|>pergunta\n<|think|>raciocínio passo a passo\n<|answer|>resposta final\x00

Como alimentar o treino com seus próprios dados CoT (qualquer um destes, na pasta do script):
    • .jsonl  -> {"question": "...", "reasoning": "...", "answer": "..."}   (também: prompt/input, cot/thought, output/response)
    • .csv    -> colunas com esses mesmos nomes
    • .parquet -> idem (precisa de pyarrow: pip install pyarrow)
    • Alpaca (instruction/input/output) em csv/jsonl/parquet: se o output trouxer o raciocínio
      dentro de tags (<thinking>..</thinking><answer>..</answer>, <think>..</think>resp ou
      ..</think>resp), ele é separado e convertido pro formato acima automaticamente.
    • .txt    -> já no formato acima (com os tokens <|q|> <|think|> <|answer|>)

Uso:
    python ia_server.py                       # treina (offline) e encerra
    python ia_server.py --serve [--wiki] [--training-local] [--port 5000]
    python ia_server.py --ask "What is 23 + 48?" [--show-thoughts] [--no-cot]
    python ia_server.py --ask "Hello!" --chat   # modo conversa (<|chat|>)
    python ia_server.py --max-minutes 120     # combina com qualquer modo

Variáveis de ambiente: MODEL_PRESET, CONTEXT_LEN, BATCH_SIZE, BPE_VOCAB_SIZE,
COT_SYNTH_EXAMPLES, COT_SYNTH_LANG (en|pt), COT_BATCH_FRACTION, SAVE_OPT_STATE,
CHAT_SYNTH_EXAMPLES, CHAT_SYNTH_LANG (en|pt|both), CHAT_BATCH_FRACTION, REUSE_LEGACY_TEXT_CACHE.
Requer PyTorch >= 2.0 (usa scaled_dot_product_attention).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time, random, math, os, threading, csv, signal, heapq, array, contextlib, hashlib
import re as _re, json as _json
from collections import deque
from rich.console import Console
from rich.markup import escape
try:
    import requests
except ImportError:
    requests = None
try:
    from flask import Flask, request, jsonify, send_file
except ImportError:
    Flask = None
console = Console()
_server_lock = threading.RLock()
SEED = 42

# ═══════════════════════════════════════════════════════════════════════
# HIPERPARÂMETROS
# ═══════════════════════════════════════════════════════════════════════
CONTEXT_LEN = int(os.environ.get('CONTEXT_LEN', 1024))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 6))
PRETRAIN_STEPS = 50000  # teto de segurança — o treino normalmente para antes, por loss (abaixo)
# --- parada por loss — medida SÓ nas linhas de texto do batch (as linhas CoT ficam de fora,
#     senão a loss baixa dos exemplos sintéticos "enganaria" o critério e pararia cedo) ---
LOSS_WINDOW = 200
LOSS_MIN_STEPS = 1000
LOSS_TARGET = 3.3
LOSS_PLATEAU_PATIENCE = 2000
LOSS_PLATEAU_MIN_DELTA = 0.01
PRETRAIN_LR = 0.0006
PRETRAIN_LR_CONT = 0.0003
PRETRAIN_WARMUP_STEPS = 100
# --- treino local por página (--training-local) ---
LOCAL_TRAIN_LOSS_TARGET = 2.5
LOCAL_TRAIN_MAX_STEPS_POR_PAGINA = 300
REPLAY_BUFFER_MAX_PAGES = 30
REPLAY_BATCH_SIZE = 4
REPLAY_FRACTION = 0.5
# --- CoT ---
COT_SYNTH_EXAMPLES = int(os.environ.get('COT_SYNTH_EXAMPLES', 5000))  # 0 desliga o gerador sintético
COT_SYNTH_LANG = os.environ.get('COT_SYNTH_LANG', 'en')                # 'en' ou 'pt'
COT_BATCH_FRACTION = float(os.environ.get('COT_BATCH_FRACTION', 0.10)) # fração das linhas do batch vindas de dados CoT
COT_MAX_THINK_TOKENS = 256    # teto de tokens de "pensamento" na geração
COT_SAMPLE_EVERY = 500        # a cada N steps de treino, imprime um exemplo de CoT/chat gerado (0 desliga)
# --- Chat (task token <|chat|>) ---
CHAT_SYNTH_EXAMPLES = int(os.environ.get('CHAT_SYNTH_EXAMPLES', 8000))  # conversas sintéticas; 0 desliga
CHAT_SYNTH_LANG = os.environ.get('CHAT_SYNTH_LANG', 'both')             # 'en' | 'pt' | 'both'
CHAT_BATCH_FRACTION = float(os.environ.get('CHAT_BATCH_FRACTION', 0.05))  # fração das linhas do batch vindas de chat
if CHAT_SYNTH_LANG not in ('en', 'pt', 'both'):
    CHAT_SYNTH_LANG = 'both'
GEN_TOP_K = 40

# --- modelo ---
MODEL_PRESETS = {
    'tiny':  dict(embed_dim=256, n_heads=4, n_layers=6,  dropout=0.1),  # ~6M params  | ckpt ~76MB  | fp16 ~13MB
    'small': dict(embed_dim=384, n_heads=6, n_layers=8,  dropout=0.1),  # ~17M params | ckpt ~200MB | fp16 ~33MB
    'base':  dict(embed_dim=512, n_heads=8, n_layers=10, dropout=0.1),  # ~35M params | ckpt ~416MB | fp16 ~69MB
}
MODEL_PRESET = os.environ.get('MODEL_PRESET', 'small')
if MODEL_PRESET not in MODEL_PRESETS:
    MODEL_PRESET = 'small'
INIT_CONFIG = dict(MODEL_PRESETS[MODEL_PRESET])

# --- checkpoints ---
_KAGGLE_WORKING = '/kaggle/working'
if os.path.isdir(_KAGGLE_WORKING):
    CHECKPOINT_DIR = os.path.join(_KAGGLE_WORKING, 'checkpoints')
else:
    CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', CHECKPOINT_DIR)   # ex.: /data/checkpoints num HF Space
CHECKPOINT_EVERY = 50
CHECKPOINT_LAST = 'checkpoint_last.pt'
WEIGHTS_EXPORT = 'model_fp16.pt'
CKPT_FORMAT = 2   # 1 = formato antigo (MoE/3 otimizadores/máscara salva) — incompatível
SAVE_OPT_STATE = os.environ.get('SAVE_OPT_STATE', '1') != '0'  # 0 = checkpoint só com pesos (não retoma o Adam)

torch.manual_seed(SEED)
random.seed(SEED)
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

CSV_MAX_CHARS_POR_ARQUIVO = 2000000
CSV_SAMPLE_LINHAS = 200
CSV_MIN_LEN_MEDIO_TEXTO = 15
CORPUS_MAX_FILES = 50
# Limites pensados pra CPU: o BPE e o encode() são Python puro (~330 bytes de RAM por caractere
# no treino do BPE), então corpus de centenas de MB não terminam nunca / estouram a RAM.
CORPUS_MAX_CHARS_TOTAL = int(os.environ.get('CORPUS_MAX_CHARS_TOTAL', 6000000000))
TXT_MAX_CHARS_POR_ARQUIVO = int(os.environ.get('TXT_MAX_CHARS_POR_ARQUIVO', 2000000000))  # .txt: lê só o começo
COT_MAX_CHARS_POR_ARQUIVO = int(os.environ.get('COT_MAX_CHARS_POR_ARQUIVO', 10000000))  # csv/jsonl/parquet com CoT
BPE_TRAIN_MAX_CHARS = int(os.environ.get('BPE_TRAIN_MAX_CHARS', 3000000))   # amostra usada só pra treinar o tokenizer
IGNORE_FILES = {'requirements.txt', 'readme.txt', 'license.txt', 'robots.txt'}
IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.env', 'dist', 'build', '.pytest_cache', '.tox', '.idea', '.vscode'}

EOS = '\x00'

# ═══════════════════════════════════════════════════════════════════════
# CoT — tokens especiais, formato e gerador sintético
# ═══════════════════════════════════════════════════════════════════════
# Os tokens especiais NÃO passam pelo BPE: ganham ids fixos logo depois do vocab
# do BPE (ver _set_tokenizer), então são sempre atômicos e nunca "vazam" pra merges.
Q_TOKEN = '<|q|>'
THINK_TOKEN = '<|think|>'
ANSWER_TOKEN = '<|answer|>'
# Task tokens (escolhem o modo) e marcadores de papel do chat. A ORDEM importa: os 3 primeiros mantêm os
# ids antigos, então checkpoints anteriores são transplantados (só as linhas novas do embedding são aleatórias).
CHAT_TOKEN = '<|chat|>'
COT_TOKEN = '<|cot|>'
USER_TOKEN = '<|user|>'
ASSISTANT_TOKEN = '<|assistant|>'
SPECIAL_TOKENS = (Q_TOKEN, THINK_TOKEN, ANSWER_TOKEN, CHAT_TOKEN, COT_TOKEN, USER_TOKEN, ASSISTANT_TOKEN)
_SPECIAL_RE = _re.compile('(' + '|'.join(_re.escape(t) for t in SPECIAL_TOKENS) + ')')


def _formatar_cot(pergunta: str, raciocinio: str, resposta: str) -> str:
    return f"{COT_TOKEN}{Q_TOKEN}{pergunta.strip()}\n{THINK_TOKEN}{raciocinio.strip()}\n{ANSWER_TOKEN}{resposta.strip()}{EOS}"


def _normalizar_cot_txt(texto: str) -> str:
    """.txt CoT antigo (sem <|cot|>): põe o task token na frente de cada <|q|> que ainda não tem."""
    return _re.sub('(?<!' + _re.escape(COT_TOKEN) + ')' + _re.escape(Q_TOKEN), COT_TOKEN + Q_TOKEN, texto)


_COT_TXT = {
    'en': {
        'soma_q': 'What is {a} + {b}?',
        'soma_r': '{a} + {b}. Add the tens: {ta} + {tb} = {t}. Add the ones: {ua} + {ub} = {u}. Then {t} + {u} = {r}.',
        'sub_q': 'What is {a} - {b}?',
        'sub_r': '{a} - {b}. Subtract the tens: {a} - {tb} = {m}. Subtract the ones: {m} - {ub} = {r}.',
        'mul_q': 'What is {a} x {b}?',
        'mul_r': 'Split {b} into {tb} + {ub}. {a} x {tb} = {p1}. {a} x {ub} = {p2}. Then {p1} + {p2} = {r}.',
        'seq_q': 'What number comes next: {s0}, {s1}, {s2}, {s3}, ?',
        'seq_r': 'The step between terms is {d}. The last term is {s3}. {s3} + {d} = {r}.',
        'par_q': 'Is {n} even or odd?',
        'par_r': 'Look at the last digit of {n}, which is {u}. {u} is {w}, so {n} is {w}.',
        'par_even': 'even', 'par_odd': 'odd',
        'cmp_q': 'Which is larger, {a} or {b}?',
        'cmp_r': 'Compare {a} and {b}: {a} - {b} = {d}. The difference is {sinal}, so {r} is larger.',
        'cmp_pos': 'positive', 'cmp_neg': 'negative',
        'loja_q': '{nome} has {a} apples. {nome} buys {b} more and gives away {c}. How many apples are left?',
        'loja_r': 'Start with {a}. After buying {b} more: {a} + {b} = {s}. After giving away {c}: {s} - {c} = {r}.',
    },
    'pt': {
        'soma_q': 'Quanto é {a} + {b}?',
        'soma_r': '{a} + {b}. Some as dezenas: {ta} + {tb} = {t}. Some as unidades: {ua} + {ub} = {u}. Então {t} + {u} = {r}.',
        'sub_q': 'Quanto é {a} - {b}?',
        'sub_r': '{a} - {b}. Tire as dezenas: {a} - {tb} = {m}. Tire as unidades: {m} - {ub} = {r}.',
        'mul_q': 'Quanto é {a} x {b}?',
        'mul_r': 'Separe {b} em {tb} + {ub}. {a} x {tb} = {p1}. {a} x {ub} = {p2}. Então {p1} + {p2} = {r}.',
        'seq_q': 'Qual número vem a seguir: {s0}, {s1}, {s2}, {s3}, ?',
        'seq_r': 'O passo entre os termos é {d}. O último termo é {s3}. {s3} + {d} = {r}.',
        'par_q': '{n} é par ou ímpar?',
        'par_r': 'Olhe o último dígito de {n}, que é {u}. {u} é {w}, então {n} é {w}.',
        'par_even': 'par', 'par_odd': 'ímpar',
        'cmp_q': 'Qual é maior, {a} ou {b}?',
        'cmp_r': 'Compare {a} e {b}: {a} - {b} = {d}. A diferença é {sinal}, então {r} é maior.',
        'cmp_pos': 'positiva', 'cmp_neg': 'negativa',
        'loja_q': '{nome} tem {a} maçãs. {nome} compra mais {b} e dá {c}. Quantas maçãs sobram?',
        'loja_r': 'Comece com {a}. Depois de comprar mais {b}: {a} + {b} = {s}. Depois de dar {c}: {s} - {c} = {r}.',
    },
}
if COT_SYNTH_LANG not in _COT_TXT:
    COT_SYNTH_LANG = 'en'
_COT_NOMES = ('Ana', 'Tom', 'Lia', 'Pedro', 'Sam', 'Maya', 'Leo')


def _gerar_cot_sintetico(n: int, lang: str = 'en', seed: int = 1234) -> str:
    """Gera n exemplos de raciocínio passo a passo (aritmética, sequências, paridade,
    comparação, problemas de palavra). Serve pra o modelo aprender o FORMATO do CoT
    e a disciplina de decompor antes de responder. Usa um Random local: não mexe no
    estado do random global."""
    rng = random.Random(seed)
    L = _COT_TXT[lang]
    out = []
    for _ in range(n):
        tipo = rng.choice(('soma', 'sub', 'mul', 'seq', 'par', 'cmp', 'loja'))
        if tipo == 'soma':
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            ta, tb, ua, ub = a // 10 * 10, b // 10 * 10, a % 10, b % 10
            q = L['soma_q'].format(a=a, b=b)
            r = L['soma_r'].format(a=a, b=b, ta=ta, tb=tb, t=ta + tb, ua=ua, ub=ub, u=ua + ub, r=a + b)
            ans = a + b
        elif tipo == 'sub':
            a = rng.randint(20, 99)
            b = rng.randint(10, a - 1)
            tb, ub = b // 10 * 10, b % 10
            q = L['sub_q'].format(a=a, b=b)
            r = L['sub_r'].format(a=a, b=b, tb=tb, ub=ub, m=a - tb, r=a - b)
            ans = a - b
        elif tipo == 'mul':
            a, b = rng.randint(2, 9), rng.randint(12, 99)
            tb, ub = b // 10 * 10, b % 10
            q = L['mul_q'].format(a=a, b=b)
            r = L['mul_r'].format(a=a, b=b, tb=tb, ub=ub, p1=a * tb, p2=a * ub, r=a * b)
            ans = a * b
        elif tipo == 'seq':
            s, d = rng.randint(1, 50), rng.randint(2, 12)
            t = [s + i * d for i in range(4)]
            q = L['seq_q'].format(s0=t[0], s1=t[1], s2=t[2], s3=t[3])
            r = L['seq_r'].format(d=d, s3=t[3], r=t[3] + d)
            ans = t[3] + d
        elif tipo == 'par':
            nn_ = rng.randint(10, 9999)
            u = nn_ % 10
            w = L['par_even'] if u % 2 == 0 else L['par_odd']
            q = L['par_q'].format(n=nn_)
            r = L['par_r'].format(n=nn_, u=u, w=w)
            ans = w
        elif tipo == 'cmp':
            a = rng.randint(100, 999)
            b = rng.randint(100, 999)
            while b == a:
                b = rng.randint(100, 999)
            d = a - b
            q = L['cmp_q'].format(a=a, b=b)
            r = L['cmp_r'].format(a=a, b=b, d=d, sinal=L['cmp_pos'] if d > 0 else L['cmp_neg'], r=max(a, b))
            ans = max(a, b)
        else:
            a, b = rng.randint(5, 60), rng.randint(1, 40)
            c = rng.randint(1, a + b)
            nome = rng.choice(_COT_NOMES)
            q = L['loja_q'].format(nome=nome, a=a, b=b, c=c)
            r = L['loja_r'].format(a=a, b=b, c=c, s=a + b, r=a + b - c)
            ans = a + b - c
        out.append(_formatar_cot(q, r, str(ans)))
    return ''.join(out)


# ═══════════════════════════════════════════════════════════════════════
# CHAT — task token <|chat|>, formato e gerador sintético
# ═══════════════════════════════════════════════════════════════════════
# Formato (um único <|chat|> no começo muda o modo; os papéis são tokens atômicos, igual ao <|think|>
# do CoT — assim o fim do prompt é sempre um token especial e não existe divergência de tokenização
# entre treino e geração):
#     <|chat|><|user|>Hello!\n<|assistant|>Hi! How can I help?\x00
# Multi-turno (cada resposta intermediária termina em \n, a última em EOS):
#     <|chat|><|user|>Hi\n<|assistant|>Hello!\n<|user|>Your name?\n<|assistant|>I'm TinyAI.\x00
def _formatar_chat(turnos) -> str:
    """turnos = [(user, assistant), ...] -> texto de treino de UMA conversa."""
    partes = [CHAT_TOKEN]
    for i, (u, a) in enumerate(turnos):
        fim = EOS if i == len(turnos) - 1 else '\n'
        partes.append(f"{USER_TOKEN}{u.strip()}\n{ASSISTANT_TOKEN}{a.strip()}{fim}")
    return ''.join(partes)


def _limpar_entrada(s: str) -> str:
    """Texto digitado pelo usuário: tira tokens especiais e EOS, pra ele não conseguir forjar a estrutura."""
    return _SPECIAL_RE.sub('', s or '').replace(EOS, '')


def _montar_prompt_chat(historico, mensagem: str) -> str:
    """<|chat|> + turnos anteriores + mensagem atual + <|assistant|> (o modelo continua a partir daqui)."""
    partes = [CHAT_TOKEN]
    for u, a in historico or []:
        partes.append(f"{USER_TOKEN}{u.strip()}\n{ASSISTANT_TOKEN}{a.strip()}\n")
    partes.append(f"{USER_TOKEN}{mensagem.strip()}\n{ASSISTANT_TOKEN}")
    return ''.join(partes)


_CHAT_NOMES = ('Ana', 'Tom', 'Lia', 'Pedro', 'Sam', 'Maya', 'Leo', 'Maria', 'John', 'Sofia', 'Lucas',
               'Carlos', 'Emma', 'Julia', 'Bruno', 'Clara')

# Categorias estáticas: nome -> (mensagens do usuário, respostas). Qualquer resposta da lista serve pra
# qualquer mensagem da mesma categoria (por isso categorias como bom-dia/boa-noite são separadas).
_CHAT_TXT = {
    'en': {
        'cats': {
            'greet': (
                ('Hello', 'Hello!', 'Hi', 'Hi!', 'Hi there', 'Hey', 'Hey!', 'Hey there', 'Hello there', 'Greetings',
                 'Howdy', 'Hiya', 'Heya', 'Hi TinyAI', 'Hello TinyAI', 'Hey TinyAI', 'Hello, friend',
                 'Hi, nice to see you', 'Good to see you', 'Yo', 'Hello everyone'),
                ('Hello! How can I help you today?', 'Hi there! What can I do for you?',
                 'Hey! Nice to see you. What is on your mind?', 'Hello! How can I help?',
                 'Hi! Ask me anything you like.', 'Hey there! How can I help you today?',
                 "Hello! I'm TinyAI. What would you like to talk about?",
                 "Hi! I'm here to help. What do you need?", 'Greetings! What can I do for you?',
                 "Hello! It's nice to hear from you.", 'Hi! Great to see you. How can I help?',
                 'Hello! What would you like to do today?', 'Hey! I am ready to chat. What is up?',
                 'Hi there! Feel free to ask me anything.')),
            'morning': (
                ('Good morning', 'Good morning!', 'Morning!', 'Good morning, TinyAI', 'Morning',
                 'Top of the morning to you'),
                ('Good morning! I hope you have a great day. How can I help?',
                 'Good morning! What can I do for you today?', 'Morning! Ready to help. What is on your mind?',
                 'Good morning! I hope you slept well. How can I help?',
                 'Good morning! Let us make today a good one. What do you need?')),
            'afternoon': (
                ('Good afternoon', 'Good afternoon!', 'Good afternoon, TinyAI', 'Afternoon!'),
                ('Good afternoon! How can I help you?', 'Good afternoon! What can I do for you?',
                 'Good afternoon! I hope your day is going well. What do you need?', 'Afternoon! How can I help?')),
            'evening': (
                ('Good evening', 'Good evening!', 'Good evening, TinyAI', 'Evening!'),
                ('Good evening! How can I help you?', 'Good evening! What can I do for you?',
                 'Good evening! I hope you had a nice day. What do you need?', 'Evening! How can I help?')),
            'night': (
                ('Good night', 'Good night!', 'Night night', 'Goodnight', 'Good night, TinyAI',
                 "I'm going to sleep now, good night"),
                ('Good night! Sleep well.', 'Good night! Sweet dreams.', 'Sleep well! See you next time.',
                 'Good night! Rest well and take care.')),
            'howru': (
                ('How are you?', 'How are you doing?', "How's it going?", 'How are you today?', 'How have you been?',
                 'Are you okay?', 'Hello, how are you?', 'Hi! How are you?', 'Hey, how are you?',
                 'How do you feel today?', "What's up?", 'Are you doing well?'),
                ("I'm doing well, thank you! How about you?", "I'm great, thanks for asking! How are you?",
                 'Doing fine! How are you doing today?', "I'm good! How can I help you today?",
                 "I'm just a small AI, but I'm doing well. How are you?", 'All good here! What about you?',
                 'Pretty good, thanks! What can I do for you?', "I'm doing great. Thanks for asking!")),
            'fine': (
                ("I'm fine", "I'm good", "I'm great, thanks", 'Doing well', 'Not bad', 'Pretty good', "I'm okay",
                 'Fine, thanks!', "I'm doing great", 'Good, thank you', 'I am well, thanks', 'All good'),
                ('Glad to hear that! What would you like to talk about?', "That's great! How can I help you?",
                 'Good to hear! What can I do for you?', 'Nice! Is there anything I can help you with?',
                 "I'm glad you're doing well. What's on your mind?")),
            'sad': (
                ("I'm sad", 'I feel sad', 'I had a bad day', "I'm not feeling well", 'I feel down', "I'm stressed",
                 "I'm upset", 'I feel lonely', "I'm having a rough day", 'Today was terrible'),
                ("I'm sorry to hear that. Do you want to talk about it?",
                 "That sounds hard. I'm here if you want to talk.",
                 "I'm sorry you're feeling this way. Take a deep breath. What happened?",
                 "I'm here to listen. What's on your mind?",
                 "I'm sorry. It's okay to feel that way. Do you want to tell me more?",
                 "That's tough. I'm here for you. What happened today?")),
            'bored': (
                ("I'm bored", 'I am so bored', 'I have nothing to do', 'Entertain me', 'Say something interesting'),
                ("Let's fix that! I can tell you a joke or we can chat. What sounds good?",
                 'Want to hear a joke? Just ask, or we can talk about anything you like.',
                 'How about a joke, a fun fact, or a quick math puzzle? You choose.',
                 'I can tell you a joke or we can play with numbers. What do you prefer?')),
            'name': (
                ('What is your name?', "What's your name?", 'Who are you?', 'Who am I talking to?',
                 'Tell me your name', 'What are you called?', 'Introduce yourself', 'Can you introduce yourself?',
                 'What should I call you?', 'Do you have a name?'),
                ("I'm TinyAI, a small language model. Nice to meet you!",
                 "My name is TinyAI. I'm a small AI that likes to chat and solve simple problems.",
                 "I'm TinyAI, a tiny AI assistant. How can I help you?",
                 'You can call me TinyAI. I am a small language model.',
                 "I'm TinyAI! I'm a small AI that can chat and do simple math step by step.",
                 "Hi, I'm TinyAI, a small AI assistant. What can I do for you?")),
            'human_q': (
                ('Are you human?', 'Are you a person?', 'Are you a real person?', 'Am I talking to a human?'),
                ("No, I'm not human. I'm TinyAI, a small AI language model.",
                 "I'm not a person. I'm an AI called TinyAI.",
                 "No, I'm an AI. I'm a small language model named TinyAI.",
                 "I'm an AI, not a human, but I'm happy to chat with you.")),
            'ai_q': (
                ('Are you an AI?', 'Are you a bot?', 'Are you an artificial intelligence?',
                 'Are you a language model?', 'Are you a chatbot?', 'Are you a computer program?'),
                ("Yes, I'm an AI. I'm a small language model called TinyAI.",
                 "Yes! I'm TinyAI, a small artificial intelligence.",
                 "That's right. I'm an AI language model, a small one named TinyAI.",
                 "Yes, I am. I'm a small AI that was trained to chat and solve simple problems.")),
            'creator': (
                ('Who made you?', 'Who created you?', 'Who built you?', 'Who is your creator?', 'Who trained you?',
                 'Who programmed you?'),
                ("I'm TinyAI, a small language model trained from scratch by my developer.",
                 'I was trained by my developer as a small experimental language model.',
                 "A developer trained me from scratch. I'm a small model, so I don't know much more than that.",
                 "I'm a small language model called TinyAI, built and trained by my developer.")),
            'can_do': (
                ('What can you do?', 'How can you help me?', 'What are you good at?', 'What do you do?',
                 'What can I ask you?', 'What are your abilities?', 'Help', 'What are you for?'),
                ('I can chat with you, answer simple questions, and solve small math problems step by step. '
                 'What would you like to try?',
                 'I can talk with you, tell jokes, and work through simple math one step at a time.',
                 "I'm a small model, so I'm best at simple chats and basic math. Try me!",
                 'I can have a friendly conversation, answer easy questions, and show my reasoning on math problems.',
                 'Ask me anything simple! I can chat, tell a joke, or do some math.')),
            'no_realtime': (
                ("What's the weather today?", "What's the weather like?", 'Will it rain today?', 'What time is it?',
                 'What time is it now?', "What's the date today?", 'What day is it today?', "What's the news today?",
                 'What is the latest news?', 'Who won the game last night?', 'What is the temperature outside?',
                 'Is it sunny outside?'),
                ("I can't check that because I don't have access to real-time information. "
                 "Your phone or a search engine will help.",
                 "Sorry, I don't have a clock, calendar, or internet connection, so I can't tell you that.",
                 "I can't look that up. I only know what I learned during training, not what is happening right now.",
                 "I don't have access to live information, so I can't answer that. "
                 "A weather app or a news site is a better choice.",
                 "That needs live information, and I don't have it. Sorry! Can I help with something else?")),
            'browse': (
                ('Can you browse the internet?', 'Can you search the web?', 'Do you have internet access?',
                 'Can you google that?', 'Are you connected to the internet?', 'Can you look things up online?'),
                ("No, I can't browse the internet. I only know what I learned during training.",
                 "No, I don't have internet access. I answer from what I learned while I was trained.",
                 "I can't search the web. I'm a small offline language model.",
                 "No, I can't look things up online, but I'll do my best to help with what I know.")),
            'thanks': (
                ('Thanks', 'Thanks!', 'Thank you', 'Thank you!', 'Thanks a lot', 'Thank you so much', 'Thx',
                 'Many thanks', 'Great, thanks!', 'Thank you, that helped', 'Thanks, TinyAI', 'Appreciate it'),
                ("You're welcome!", 'Happy to help!', 'Anytime! Let me know if you need anything else.',
                 'No problem! Is there anything else I can do?', 'My pleasure!', 'Glad I could help!',
                 "You're welcome! Come back whenever you like.", "Of course! I'm here if you need more help.")),
            'bye': (
                ('Bye', 'Bye!', 'Goodbye', 'Goodbye!', 'See you later', 'See you', 'Talk to you later',
                 'I have to go', 'Bye bye', 'Take care', 'Gotta go', 'Farewell', 'Catch you later',
                 'See you tomorrow', 'I must leave now'),
                ('Goodbye! Have a great day!', 'See you later! Take care.', 'Bye! It was nice talking to you.',
                 'Take care! Come back anytime.', 'Goodbye! I hope to talk to you again soon.',
                 'Bye for now! Have a wonderful day.', 'See you! Thanks for chatting with me.')),
            'compliment': (
                ("You're smart", "You're great", 'Good job', 'Well done', "You're awesome", 'I like you',
                 "You're helpful", 'Great answer', 'You are amazing', 'Nice work', 'That was great'),
                ("Thank you! That's kind of you to say.", "Thanks! I'm glad I could help.",
                 "That's very kind. Thank you!", "I appreciate it! I'm still learning, but I try my best.",
                 'Thank you so much! That made my day.', 'Aw, thanks! Happy to help anytime.')),
            'wrong': (
                ("That's wrong", 'That is not correct', "You're wrong", "That's not right", 'You made a mistake',
                 'No, that is incorrect', 'That answer is wrong'),
                ("I'm sorry about that. I'm a small model and I do make mistakes. Can you tell me what was wrong?",
                 "I might be wrong, since I'm a small model. Could you tell me what the right answer is?",
                 "Sorry! I'm still learning. Can you explain what is wrong so I can try again?",
                 'My apologies. Let me try again if you give me a little more detail.')),
            'rude': (
                ("You're stupid", "You're useless", 'You are dumb', 'I hate you', 'You are so bad', 'This is stupid'),
                ("I'm sorry I let you down. I'm a small model and I'm still learning. Can I try again?",
                 "I'm sorry you feel that way. I'll do my best to do better. What do you need?",
                 "I'm sorry I disappointed you. Tell me what you need and I'll try again.")),
            'help': (
                ('Can you help me?', 'I need help', 'Help me please', 'Could you help me with something?',
                 'I have a question', 'Can I ask you something?', 'Can I ask a question?', 'I need your help',
                 'Please help me'),
                ('Of course! What do you need help with?', 'Sure! Go ahead and ask.', "Absolutely. What's your question?",
                 "Yes, I'd be happy to help. What is it?", "I'm listening. What do you need?",
                 'Sure thing! Tell me what is going on.')),
            'present': (
                ('Are you there?', 'Can you hear me?', 'Hello? Are you there?', 'Are you listening?',
                 'Is anyone there?', 'Are you still there?'),
                ("Yes, I'm here! What can I do for you?", "I'm here and listening. Go ahead.",
                 "Yes! I'm here. How can I help?", "I'm here. What's up?")),
            'joke': (
                ('Tell me a joke', 'Say something funny', 'Do you know any jokes?', 'Make me laugh',
                 'Another joke, please', 'Got any jokes?', 'Tell me another joke', 'Can you tell me a joke?'),
                ("Why don't scientists trust atoms? Because they make up everything!",
                 'Why did the computer go to the doctor? Because it had a virus!',
                 'What do you call a fish with no eyes? A fsh!',
                 'Why was the math book sad? It had too many problems.',
                 'Why do programmers prefer dark mode? Because light attracts bugs!',
                 'What do you call a sleeping dinosaur? A dino-snore!',
                 'Why did the scarecrow win an award? Because he was outstanding in his field!',
                 'What did the ocean say to the beach? Nothing, it just waved.',
                 'How does a penguin build its house? Igloos it together!')),
            'feelings': (
                ('Do you have feelings?', 'Do you get sad?', 'Do you get tired?', 'Do you ever feel lonely?',
                 'Can you feel emotions?'),
                ("I'm an AI, so I don't have feelings like people do, but I enjoy chatting with you.",
                 "No, I don't feel emotions the way people do. I'm just a program, but I'm happy to talk.",
                 "I don't get tired or sad because I'm an AI, but I'm glad to keep you company.",
                 "As an AI, I don't have real feelings, but I'm always ready to help.")),
            'favorite': (
                ("What's your favorite color?", "What's your favorite food?", "What's your favorite animal?",
                 'Do you have a favorite song?', 'What is your favorite movie?', 'What do you like to do?'),
                ("I'm an AI, so I don't really have favorites, but I'm curious. What's yours?",
                 "I don't have personal favorites, but I'd love to hear about yours.",
                 "As an AI, I don't have favorites. What about you?")),
            'age': (
                ('How old are you?', 'What is your age?', 'When were you born?'),
                ("I'm a young model! I don't have an age like people do.",
                 "I don't have a birthday like people do. I'm a small AI that was trained not long ago.",
                 "I'm an AI, so I don't age. I'm just a young, small language model.")),
            'where': (
                ('Where are you from?', 'Where do you live?', 'Where are you?'),
                ("I live inside a computer. I'm just a program!",
                 "I don't live anywhere in the way people do. I run on a computer.",
                 "I'm a program, so I live on a computer or server somewhere.")),
            'language': (
                ('Do you speak English?', 'Do you speak Portuguese?', 'What languages do you speak?',
                 'Can you speak other languages?', 'Which language do you speak?'),
                ("I chat mostly in English, but I'm learning other languages too.",
                 "I speak mainly English, though I'm still learning other languages.",
                 "English is my strongest language, but I'm learning more. I make mistakes, so please be patient!")),
            'ack': (
                ('ok', 'okay', 'alright', 'got it', 'I see', 'cool', 'nice', 'great', 'understood', 'sure'),
                ('Great! Is there anything else I can help with?', 'Okay! Let me know if you need anything else.',
                 "Glad we're on the same page. Anything else?", 'Alright! I am here if you need me.',
                 'Cool! What would you like to do next?')),
            'gibberish': (
                ('asdf', 'qwerty', '???', 'hmm', 'uh', '...', 'lol', 'huh?', 'hmmmm', 'blah', 'aaa'),
                ("I didn't quite understand that. Could you say it in a different way?",
                 "Sorry, I'm not sure what you mean. Can you rephrase?",
                 "Hmm, I didn't get that. What would you like to talk about?",
                 'I could not understand that. Could you try again?')),
            'unknown': (
                ('What is the meaning of life?', 'Will I be rich?', 'Who will win the next election?',
                 'What will happen tomorrow?', 'Can you predict the future?'),
                ("That's a big question, and I'm not sure. I'm a small model, so I don't have a good answer.",
                 "I can't predict the future. I can only share what I learned during training.",
                 "I'm not sure about that. It might be better to ask an expert or look it up.")),
            'reasoning_q': (
                ('Can you think step by step?', 'Can you show your reasoning?', 'Do you think before answering?'),
                ('Yes! In reasoning mode I work through problems step by step before giving the final answer.',
                 'I can. In reasoning mode I write my steps first and then give the final answer.')),
        },
        # (mensagens, respostas) com {n} = nome do usuário
        'name_intro': (
            ('My name is {n}', "I'm {n}", 'I am {n}', 'Call me {n}', "Hi, I'm {n}", 'Hello, my name is {n}',
             'You can call me {n}'),
            ('Nice to meet you, {n}! How can I help you?', "Hello, {n}! It's nice to meet you.",
             'Nice to meet you, {n}!', 'Hi {n}! What can I do for you today?', 'Welcome, {n}! How can I help?')),
        'recall': (
            ('What is my name?', "What's my name?", 'Do you remember my name?', 'Who am I?',
             'Can you tell me my name?'),
            ("I don't know your name yet. What should I call you?", "You haven't told me your name yet. What is it?",
             "I don't think you told me your name. What is it?")),
        'recall_known': ('Your name is {n}.', 'You told me your name is {n}.', "You're {n}!",
                         'Your name is {n}. Nice to talk with you!'),
        'math_frames': ('What is {e}?', 'How much is {e}?', 'Calculate {e}', '{e}', '{e} = ?', 'Compute {e}',
                        "What's {e}?"),
        'math_ops': {'+': ('{a} + {b}', '{a} plus {b}'), '-': ('{a} - {b}', '{a} minus {b}'),
                     'x': ('{a} x {b}', '{a} times {b}', '{a} * {b}')},
        'math_replies': ('{e} = {r}.', 'The answer is {r}.', '{e} is {r}.', 'That is {r}.', "It's {r}.", '{r}'),
        'cap_users': ('What is the capital of {c}?', "What's the capital of {c}?", 'Capital of {c}?',
                      'Tell me the capital of {c}', 'Which city is the capital of {c}?'),
        'cap_replies': ('The capital of {c} is {k}.', '{k} is the capital of {c}.', "It's {k}.", '{k}.'),
        'capitals': (('France', 'Paris'), ('Italy', 'Rome'), ('Spain', 'Madrid'), ('Germany', 'Berlin'),
                     ('Japan', 'Tokyo'), ('Brazil', 'Brasília'), ('Portugal', 'Lisbon'),
                     ('the United Kingdom', 'London'), ('Canada', 'Ottawa'), ('Argentina', 'Buenos Aires'),
                     ('Egypt', 'Cairo'), ('Mexico', 'Mexico City'), ('Australia', 'Canberra'),
                     ('China', 'Beijing'), ('Russia', 'Moscow'), ('India', 'New Delhi'), ('Chile', 'Santiago'),
                     ('Peru', 'Lima'), ('Greece', 'Athens'), ('Ireland', 'Dublin'),
                     ('the United States', 'Washington, D.C.')),
        'facts': (
            ('How many days are in a week?', 'There are 7 days in a week.'),
            ('How many months are in a year?', 'There are 12 months in a year.'),
            ('How many hours are in a day?', 'There are 24 hours in a day.'),
            ('How many minutes are in an hour?', 'There are 60 minutes in an hour.'),
            ('How many seconds are in a minute?', 'There are 60 seconds in a minute.'),
            ('How many days are in a year?', 'A year has 365 days, or 366 in a leap year.'),
            ('How many planets are in the Solar System?', 'There are 8 planets in the Solar System.'),
            ('How many colors are in a rainbow?', 'A rainbow has 7 colors.'),
            ('How many legs does a spider have?', 'A spider has 8 legs.'),
            ('How many legs does a dog have?', 'A dog has 4 legs.'),
            ('How many sides does a triangle have?', 'A triangle has 3 sides.'),
            ('How many sides does a square have?', 'A square has 4 sides.'),
            ('What color is the sky?', 'On a clear day, the sky is blue.'),
            ('What color is grass?', 'Grass is usually green.'),
            ('What is the largest planet?', 'Jupiter is the largest planet in the Solar System.'),
            ('What is the closest star to Earth?', 'The Sun is the closest star to Earth.'),
            ('What do bees make?', 'Bees make honey.'),
            ('What is H2O?', 'H2O is the chemical formula for water.'),
            ('What is the opposite of hot?', 'The opposite of hot is cold.'),
            ('What is the opposite of big?', 'The opposite of big is small.'),
            ('What is the biggest animal?', 'The blue whale is the biggest animal on Earth.'),
            ('What is the largest ocean?', 'The Pacific Ocean is the largest ocean.'),
            ('What day comes after Monday?', 'Tuesday comes after Monday.'),
            ('What month comes after March?', 'April comes after March.'),
            ('What is the first month of the year?', 'January is the first month of the year.'),
            ('What do we call frozen water?', 'Frozen water is called ice.'),
            ('Which planet do we live on?', 'We live on planet Earth.'),
            ('At what temperature does water boil?', 'Water boils at 100 °C at sea level.'),
            ('At what temperature does water freeze?', 'Water freezes at 0 °C.'),
            ('How many letters are in the English alphabet?', 'The English alphabet has 26 letters.'),
        ),
        'echo_users': ('Say {w}', 'Say "{w}"', 'Repeat after me: {w}', 'Repeat {w}', 'Can you say {w}?',
                       'Please say {w}'),
        'echo_replies': ('{w}', '{w}.', 'Sure! {w}', '{w}!'),
        'words': ('hello', 'banana', 'apple', 'house', 'music', 'ocean', 'sunshine', 'computer', 'friend', 'pizza',
                  'robot', 'coffee', 'garden', 'river', 'window', 'rainbow'),
    },
    'pt': {
        'cats': {
            'greet': (
                ('Olá', 'Olá!', 'Oi', 'Oi!', 'Oie', 'E aí', 'Fala aí', 'Salve', 'Opa', 'Olá, TinyAI', 'Oi, TinyAI',
                 'Olá, amigo', 'Bom te ver', 'Olá a todos'),
                ('Olá! Como posso ajudar você hoje?', 'Oi! O que posso fazer por você?',
                 'Oi, tudo bem? Em que posso ajudar?', 'Olá! Como posso ajudar?', 'Oi! Pergunte o que quiser.',
                 'Olá! Eu sou a TinyAI. Sobre o que você quer conversar?',
                 'Oi! Estou aqui para ajudar. Do que você precisa?', 'Olá! É bom falar com você.',
                 'Oi! Que bom te ver. Como posso ajudar?', 'Olá! O que você gostaria de fazer hoje?')),
            'morning': (
                ('Bom dia', 'Bom dia!', 'Bom dia, TinyAI', 'Dia!', 'Bom diaa'),
                ('Bom dia! Espero que você tenha um ótimo dia. Como posso ajudar?',
                 'Bom dia! O que posso fazer por você hoje?', 'Bom dia! Estou aqui para ajudar. O que você precisa?',
                 'Bom dia! Espero que tenha dormido bem. Em que posso ajudar?')),
            'afternoon': (
                ('Boa tarde', 'Boa tarde!', 'Boa tarde, TinyAI'),
                ('Boa tarde! Como posso ajudar?', 'Boa tarde! O que posso fazer por você?',
                 'Boa tarde! Espero que seu dia esteja sendo bom. Do que você precisa?')),
            'evening': (
                ('Boa noite', 'Boa noite!', 'Boa noite, TinyAI'),
                ('Boa noite! Como posso ajudar?', 'Boa noite! O que posso fazer por você?',
                 'Boa noite! Espero que você tenha tido um bom dia. Do que você precisa?')),
            'night': (
                ('Vou dormir, boa noite', 'Vou dormir agora', 'Estou indo dormir', 'Boa noite, até amanhã'),
                ('Boa noite! Durma bem.', 'Boa noite! Bons sonhos.', 'Durma bem! Até a próxima.',
                 'Boa noite! Descanse bem e se cuide.')),
            'howru': (
                ('Como você está?', 'Tudo bem?', 'Como vai?', 'Como você está hoje?', 'Como tem passado?',
                 'Você está bem?', 'Olá, como você está?', 'Oi! Tudo bem?', 'Oi, como vai você?',
                 'E aí, tudo certo?', 'Como está se sentindo hoje?', 'Está tudo bem com você?'),
                ('Estou bem, e você?', 'Tudo ótimo por aqui! E com você?', 'Tudo bem por aqui! Como você está hoje?',
                 'Estou bem! Em que posso ajudar você hoje?', 'Sou só uma IA pequena, mas estou bem. E você?',
                 'Tudo certo! E você, como está?', 'Estou bem, agradeço por perguntar! E você?')),
            'fine': (
                ('Estou bem', 'Estou ótimo', 'Tudo bem', 'Tudo ótimo', 'Estou legal', 'Estou indo bem', 'Tudo certo',
                 'Estou bem, obrigado', 'Estou bem, obrigada'),
                ('Que bom! Sobre o que você quer conversar?', 'Fico feliz em saber! Como posso ajudar?',
                 'Que ótimo! Em que posso ajudar?', 'Bom saber! Posso ajudar em alguma coisa?',
                 'Que bom que você está bem. O que você tem em mente?')),
            'sad': (
                ('Estou triste', 'Me sinto triste', 'Tive um dia ruim', 'Não estou me sentindo bem',
                 'Estou desanimado', 'Estou estressado', 'Estou chateado', 'Me sinto sozinho',
                 'Estou com um dia difícil', 'Hoje foi terrível'),
                ('Sinto muito por isso. Quer conversar sobre o assunto?',
                 'Isso parece difícil. Estou aqui se você quiser conversar.',
                 'Sinto muito que você esteja assim. Respire fundo. O que aconteceu?',
                 'Estou aqui para ouvir. O que você tem em mente?',
                 'Sinto muito. Tudo bem se sentir assim. Quer me contar mais?',
                 'Que situação difícil. Estou aqui por você. O que aconteceu hoje?')),
            'bored': (
                ('Estou entediado', 'Estou muito entediado', 'Não tenho nada para fazer', 'Me entretenha',
                 'Diga algo interessante', 'Estou com tédio'),
                ('Vamos resolver isso! Posso contar uma piada ou podemos conversar. O que prefere?',
                 'Quer ouvir uma piada? É só pedir, ou podemos falar sobre o que você quiser.',
                 'Que tal uma piada, uma curiosidade ou um desafio de matemática? Você escolhe.',
                 'Posso contar uma piada ou brincar com números. O que você prefere?')),
            'name': (
                ('Qual é o seu nome?', 'Como você se chama?', 'Quem é você?', 'Com quem estou falando?',
                 'Me diga seu nome', 'Pode se apresentar?', 'Como devo chamar você?', 'Você tem nome?',
                 'Apresente-se'),
                ('Eu sou a TinyAI, um modelo de linguagem pequeno. Prazer em conhecer você!',
                 'Meu nome é TinyAI. Sou uma IA pequena que gosta de conversar e resolver problemas simples.',
                 'Sou a TinyAI, uma assistente de IA pequena. Como posso ajudar?',
                 'Pode me chamar de TinyAI. Sou um modelo de linguagem pequeno.',
                 'Eu sou a TinyAI! Sou uma IA pequena que conversa e faz contas simples passo a passo.',
                 'Oi, eu sou a TinyAI, uma pequena assistente de IA. O que posso fazer por você?')),
            'human_q': (
                ('Você é humano?', 'Você é uma pessoa?', 'Você é uma pessoa de verdade?',
                 'Estou falando com um humano?'),
                ('Não, eu não sou um ser humano. Sou a TinyAI, um modelo de linguagem de IA pequeno.',
                 'Não sou uma pessoa. Sou uma IA chamada TinyAI.',
                 'Não, sou uma IA. Sou um modelo de linguagem pequeno chamado TinyAI.',
                 'Sou uma IA, não um humano, mas fico feliz em conversar com você.')),
            'ai_q': (
                ('Você é uma IA?', 'Você é um robô?', 'Você é um bot?', 'Você é uma inteligência artificial?',
                 'Você é um modelo de linguagem?', 'Você é um chatbot?', 'Você é um programa de computador?'),
                ('Sim, sou uma IA. Sou um modelo de linguagem pequeno chamado TinyAI.',
                 'Sim! Eu sou a TinyAI, uma pequena inteligência artificial.',
                 'Isso mesmo. Sou um modelo de linguagem de IA, um bem pequeno chamado TinyAI.',
                 'Sim, sou uma IA. Meu treino foi feito para conversar e resolver problemas simples.')),
            'creator': (
                ('Quem fez você?', 'Quem criou você?', 'Quem construiu você?', 'Quem é o seu criador?',
                 'Quem treinou você?', 'Quem programou você?'),
                ('Sou a TinyAI, um modelo de linguagem pequeno treinado do zero pelo meu desenvolvedor.',
                 'Meu desenvolvedor me treinou como um pequeno modelo de linguagem experimental.',
                 'Um desenvolvedor me treinou do zero. Sou um modelo pequeno, então não sei muito mais que isso.',
                 'Sou um modelo de linguagem chamado TinyAI, criado e treinado pelo meu desenvolvedor.')),
            'can_do': (
                ('O que você pode fazer?', 'Como você pode me ajudar?', 'O que você sabe fazer?', 'O que você faz?',
                 'O que posso perguntar a você?', 'Quais são as suas habilidades?', 'Ajuda', 'Para que você serve?'),
                ('Posso conversar com você, responder perguntas simples e resolver pequenos problemas de matemática '
                 'passo a passo. O que você quer experimentar?',
                 'Posso bater papo, contar piadas e resolver contas simples, um passo de cada vez.',
                 'Sou um modelo pequeno, então me saio melhor em conversas simples e matemática básica. Pode testar!',
                 'Posso ter uma conversa amigável, responder perguntas fáceis e mostrar meu raciocínio em problemas '
                 'de matemática.',
                 'Pergunte algo simples! Posso conversar, contar uma piada ou fazer umas contas.')),
            'no_realtime': (
                ('Como está o tempo hoje?', 'Vai chover hoje?', 'Que horas são?', 'Que dia é hoje?',
                 'Qual é a data de hoje?', 'Quais são as notícias de hoje?', 'Quais são as últimas notícias?',
                 'Quem ganhou o jogo ontem?', 'Qual é a temperatura lá fora?', 'Está sol lá fora?',
                 'Qual é a previsão do tempo?'),
                ('Não consigo verificar isso porque não tenho acesso a informações em tempo real. '
                 'Seu celular ou um buscador ajudam.',
                 'Desculpe, não tenho relógio, calendário nem internet, então não posso dizer isso.',
                 'Não posso pesquisar isso. Só sei o que aprendi no treinamento, não o que está acontecendo agora.',
                 'Não tenho acesso a informações ao vivo, então não consigo responder. '
                 'Um aplicativo de clima ou um site de notícias é melhor.',
                 'Isso precisa de informação ao vivo, e eu não tenho. Desculpe! Posso ajudar com outra coisa?')),
            'browse': (
                ('Você consegue navegar na internet?', 'Você pode pesquisar na web?', 'Você tem acesso à internet?',
                 'Pode pesquisar isso no Google?', 'Você está conectado à internet?',
                 'Você consegue procurar coisas online?'),
                ('Não, não consigo navegar na internet. Só sei o que aprendi durante o treinamento.',
                 'Não, não tenho acesso à internet. Respondo com base no que aprendi no treinamento.',
                 'Não consigo pesquisar na web. Sou um modelo de linguagem pequeno e offline.',
                 'Não posso procurar coisas online, mas vou tentar ajudar com o que sei.')),
            'thanks': (
                ('Obrigado', 'Obrigada', 'Obrigado!', 'Obrigada!', 'Muito obrigado', 'Muito obrigada', 'Valeu',
                 'Valeu!', 'Brigado', 'Ótimo, obrigado!', 'Obrigado, isso ajudou', 'Agradeço', 'Obrigado, TinyAI'),
                ('De nada!', 'Fico feliz em ajudar!', 'Sempre que precisar! Me avise se precisar de mais alguma coisa.',
                 'Sem problema! Posso ajudar em mais alguma coisa?', 'Foi um prazer!', 'Que bom que pude ajudar!',
                 'De nada! Volte quando quiser.', 'Claro! Estou aqui se precisar de mais ajuda.')),
            'bye': (
                ('Tchau', 'Tchau!', 'Adeus', 'Até logo', 'Até mais', 'Falo com você depois', 'Preciso ir',
                 'Tchau tchau', 'Se cuida', 'Tenho que ir', 'Até a próxima', 'Até amanhã', 'Flw',
                 'Vou embora agora'),
                ('Tchau! Tenha um ótimo dia!', 'Até logo! Se cuide.', 'Tchau! Foi bom conversar com você.',
                 'Se cuide! Volte quando quiser.', 'Até mais! Espero falar com você de novo em breve.',
                 'Por ora é isso! Tenha um dia maravilhoso.', 'Até a próxima! Agradeço a conversa.')),
            'compliment': (
                ('Você é inteligente', 'Você é demais', 'Bom trabalho', 'Muito bem', 'Você é incrível',
                 'Eu gosto de você', 'Você é muito útil', 'Ótima resposta', 'Belo trabalho', 'Foi ótimo'),
                ('Isso é muito gentil da sua parte. Agradeço!', 'Fico feliz por ter ajudado!',
                 'Que gentileza! Agradeço muito.', 'Agradeço! Ainda estou aprendendo, mas faço o meu melhor.',
                 'Muito gentil! Isso alegrou o meu dia.', 'Ah, que bom ouvir isso! Posso ajudar sempre que precisar.')),
            'wrong': (
                ('Isso está errado', 'Isso não está correto', 'Isso não está certo', 'Você errou',
                 'Não, isso está incorreto', 'Essa resposta está errada'),
                ('Sinto muito por isso. Sou um modelo pequeno e às vezes erro. Pode me dizer o que estava errado?',
                 'Posso estar errando, já que sou um modelo pequeno. Qual é a resposta certa?',
                 'Desculpe! Ainda estou aprendendo. Pode explicar o que está errado para eu tentar de novo?',
                 'Peço desculpas. Posso tentar de novo se você me der mais detalhes.')),
            'rude': (
                ('Você é burro', 'Você é inútil', 'Você é bobo', 'Eu odeio você', 'Você é muito ruim',
                 'Isso é idiota'),
                ('Sinto muito por ter decepcionado. Sou um modelo pequeno e ainda estou aprendendo. '
                 'Posso tentar de novo?',
                 'Sinto muito que você pense assim. Vou tentar fazer melhor. Do que você precisa?',
                 'Desculpe se decepcionei. Diga o que você precisa e eu tento de novo.')),
            'help': (
                ('Pode me ajudar?', 'Preciso de ajuda', 'Me ajude, por favor', 'Você poderia me ajudar com uma coisa?',
                 'Tenho uma pergunta', 'Posso perguntar uma coisa?', 'Posso fazer uma pergunta?',
                 'Preciso da sua ajuda', 'Por favor, me ajude'),
                ('Claro! Com o que você precisa de ajuda?', 'Pode perguntar!', 'Com certeza. Qual é a sua pergunta?',
                 'Sim, com prazer. O que é?', 'Estou ouvindo. Do que você precisa?',
                 'Pode contar comigo! Me diga o que está acontecendo.')),
            'present': (
                ('Você está aí?', 'Está me ouvindo?', 'Alô? Você está aí?', 'Tem alguém aí?',
                 'Você ainda está aí?', 'Alguém aí?'),
                ('Sim, estou aqui! O que posso fazer por você?', 'Estou aqui e ouvindo. Pode falar.',
                 'Sim! Estou aqui. Como posso ajudar?', 'Estou aqui. Pois não?')),
            'joke': (
                ('Conte uma piada', 'Diga algo engraçado', 'Você conhece alguma piada?', 'Me faça rir',
                 'Outra piada, por favor', 'Tem alguma piada?', 'Conte outra piada', 'Pode contar uma piada?'),
                ('Por que o livro de matemática ficou triste? Porque tinha muitos problemas.',
                 'Por que o computador foi ao médico? Porque estava com vírus!',
                 'Qual é o animal mais antigo? A zebra, porque é em preto e branco.',
                 'O que o pato disse para a pata? Vem quá!',
                 'Por que o espantalho ganhou um prêmio? Porque ele se destacava no campo!',
                 'O que é que tem cabeça e dentes, mas não morde? O alho.',
                 'Por que os programadores preferem o modo escuro? Porque a luz atrai bugs!',
                 'O que uma onda disse para a outra? Nada, apenas acenou.')),
            'feelings': (
                ('Você tem sentimentos?', 'Você fica triste?', 'Você fica cansado?', 'Você já se sentiu sozinho?',
                 'Você sente emoções?'),
                ('Sou uma IA, então não tenho sentimentos como as pessoas, mas gosto de conversar com você.',
                 'Não sinto emoções como as pessoas. Sou só um programa, mas fico feliz em conversar.',
                 'Como sou uma IA, não me canso nem fico triste, mas gosto de fazer companhia.',
                 'Como IA, não tenho sentimentos de verdade, mas estou sempre aqui para ajudar.')),
            'favorite': (
                ('Qual é a sua cor favorita?', 'Qual é a sua comida favorita?', 'Qual é o seu animal favorito?',
                 'Você tem uma música favorita?', 'Qual é o seu filme favorito?', 'O que você gosta de fazer?'),
                ('Sou uma IA, então não tenho favoritos de verdade, mas tenho curiosidade. Qual é o seu?',
                 'Não tenho favoritos pessoais, mas adoraria saber os seus.',
                 'Como IA, não tenho favoritos. E você?')),
            'age': (
                ('Quantos anos você tem?', 'Qual é a sua idade?', 'Quando você nasceu?'),
                ('Sou um modelo novo! Não tenho idade como as pessoas.',
                 'Não tenho aniversário como as pessoas. Sou uma IA pequena, treinada há pouco tempo.',
                 'Sou uma IA, então não envelheço. Sou só um modelo de linguagem jovem e pequeno.')),
            'where': (
                ('De onde você é?', 'Onde você mora?', 'Onde você está?'),
                ('Eu moro dentro de um computador. Sou só um programa!',
                 'Não moro em lugar nenhum como as pessoas. Eu rodo em um computador.',
                 'Sou um programa, então vivo em algum computador ou servidor.')),
            'language': (
                ('Você fala português?', 'Você fala inglês?', 'Quais idiomas você fala?', 'Você fala outras línguas?',
                 'Em qual língua você fala?'),
                ('Converso principalmente em português, mas também estou aprendendo outros idiomas.',
                 'Falo principalmente português, mas ainda estou aprendendo outros idiomas.',
                 'O português é a língua em que me saio melhor, mas estou aprendendo mais. '
                 'Cometo erros, então tenha paciência!')),
            'ack': (
                ('ok', 'beleza', 'certo', 'entendi', 'legal', 'bacana', 'tá bom', 'combinado', 'ok, entendi', 'show'),
                ('Ótimo! Posso ajudar em mais alguma coisa?', 'Certo! Me avise se precisar de mais alguma coisa.',
                 'Que bom que nos entendemos. Mais alguma coisa?', 'Tudo bem! Estou aqui se precisar.',
                 'Legal! O que você quer fazer agora?')),
            'gibberish': (
                ('asdf', 'qwerty', '???', 'hmm', 'ééé', '...', 'kkk', 'hã?', 'hmmmm', 'blá', 'aaa'),
                ('Não entendi direito. Pode dizer de outro jeito?',
                 'Desculpe, não sei o que você quer dizer. Pode reformular?',
                 'Hmm, não entendi. Sobre o que você quer conversar?', 'Não consegui entender. Pode tentar de novo?')),
            'unknown': (
                ('Qual é o sentido da vida?', 'Vou ficar rico?', 'Quem vai ganhar a próxima eleição?',
                 'O que vai acontecer amanhã?', 'Você consegue prever o futuro?'),
                ('Essa é uma pergunta grande e não tenho certeza. Sou um modelo pequeno, então não tenho uma boa resposta.',
                 'Não consigo prever o futuro. Só posso compartilhar o que aprendi no treinamento.',
                 'Não tenho certeza sobre isso. Talvez seja melhor perguntar a um especialista ou pesquisar.')),
            'reasoning_q': (
                ('Você consegue pensar passo a passo?', 'Pode mostrar seu raciocínio?', 'Você pensa antes de responder?'),
                ('Sim! No modo de raciocínio, eu resolvo os problemas passo a passo antes de dar a resposta final.',
                 'Consigo. No modo de raciocínio, escrevo meus passos primeiro e depois dou a resposta final.')),
        },
        'name_intro': (
            ('Meu nome é {n}', 'Eu sou {n}', 'Me chamo {n}', 'Pode me chamar de {n}', 'Oi, eu sou {n}',
             'Olá, meu nome é {n}', 'Sou {n}'),
            ('Prazer em conhecer você, {n}! Como posso ajudar?', 'Olá, {n}! É um prazer conhecer você.',
             'Prazer, {n}!', 'Oi, {n}! O que posso fazer por você hoje?', 'Que bom te conhecer, {n}! Como posso ajudar?')),
        'recall': (
            ('Qual é o meu nome?', 'Você lembra o meu nome?', 'Quem sou eu?', 'Sabe qual é o meu nome?'),
            ('Ainda não sei o seu nome. Como devo chamar você?', 'Você ainda não me disse o seu nome. Qual é?',
             'Acho que você não me falou o seu nome. Qual é?')),
        'recall_known': ('O seu nome é {n}.', 'Você me disse que o seu nome é {n}.', 'Você é {n}!',
                         'Seu nome é {n}. Bom conversar com você!'),
        'math_frames': ('Quanto é {e}?', 'Quanto dá {e}?', 'Calcule {e}', '{e}', '{e} = ?',
                        'Qual é o resultado de {e}?'),
        'math_ops': {'+': ('{a} + {b}', '{a} mais {b}'), '-': ('{a} - {b}', '{a} menos {b}'),
                     'x': ('{a} x {b}', '{a} vezes {b}', '{a} * {b}')},
        'math_replies': ('{e} = {r}.', 'A resposta é {r}.', '{e} é {r}.', 'Dá {r}.', 'É {r}.', '{r}'),
        'cap_users': ('Qual é a capital {c}?', 'Qual a capital {c}?', 'Capital {c}?', 'Me diga a capital {c}',
                      'Qual cidade é a capital {c}?'),
        'cap_replies': ('A capital {c} é {k}.', '{k} é a capital {c}.', 'É {k}.', '{k}.'),
        'capitals': (('da França', 'Paris'), ('da Itália', 'Roma'), ('da Espanha', 'Madri'),
                     ('da Alemanha', 'Berlim'), ('do Japão', 'Tóquio'), ('do Brasil', 'Brasília'),
                     ('de Portugal', 'Lisboa'), ('do Reino Unido', 'Londres'), ('do Canadá', 'Ottawa'),
                     ('da Argentina', 'Buenos Aires'), ('do Egito', 'Cairo'), ('do México', 'Cidade do México'),
                     ('da Austrália', 'Camberra'), ('da China', 'Pequim'), ('da Rússia', 'Moscou'),
                     ('da Índia', 'Nova Délhi'), ('do Chile', 'Santiago'), ('do Peru', 'Lima'),
                     ('da Grécia', 'Atenas'), ('da Irlanda', 'Dublin'),
                     ('dos Estados Unidos', 'Washington, D.C.')),
        'facts': (
            ('Quantos dias tem uma semana?', 'Uma semana tem 7 dias.'),
            ('Quantos meses tem um ano?', 'Um ano tem 12 meses.'),
            ('Quantas horas tem um dia?', 'Um dia tem 24 horas.'),
            ('Quantos minutos tem uma hora?', 'Uma hora tem 60 minutos.'),
            ('Quantos segundos tem um minuto?', 'Um minuto tem 60 segundos.'),
            ('Quantos dias tem um ano?', 'Um ano tem 365 dias, ou 366 no ano bissexto.'),
            ('Quantos planetas tem o Sistema Solar?', 'O Sistema Solar tem 8 planetas.'),
            ('Quantas cores tem o arco-íris?', 'O arco-íris tem 7 cores.'),
            ('Quantas patas tem uma aranha?', 'Uma aranha tem 8 patas.'),
            ('Quantas patas tem um cachorro?', 'Um cachorro tem 4 patas.'),
            ('Quantos lados tem um triângulo?', 'Um triângulo tem 3 lados.'),
            ('Quantos lados tem um quadrado?', 'Um quadrado tem 4 lados.'),
            ('De que cor é o céu?', 'Em um dia claro, o céu é azul.'),
            ('De que cor é a grama?', 'A grama geralmente é verde.'),
            ('Qual é o maior planeta?', 'Júpiter é o maior planeta do Sistema Solar.'),
            ('Qual é a estrela mais próxima da Terra?', 'O Sol é a estrela mais próxima da Terra.'),
            ('O que as abelhas fazem?', 'As abelhas fazem mel.'),
            ('O que é H2O?', 'H2O é a fórmula química da água.'),
            ('Qual é o contrário de quente?', 'O contrário de quente é frio.'),
            ('Qual é o contrário de grande?', 'O contrário de grande é pequeno.'),
            ('Qual é o maior animal?', 'A baleia-azul é o maior animal da Terra.'),
            ('Qual é o maior oceano?', 'O Oceano Pacífico é o maior oceano.'),
            ('Que dia vem depois de segunda-feira?', 'Terça-feira vem depois de segunda-feira.'),
            ('Que mês vem depois de março?', 'Abril vem depois de março.'),
            ('Qual é o primeiro mês do ano?', 'Janeiro é o primeiro mês do ano.'),
            ('Como chamamos a água congelada?', 'A água congelada se chama gelo.'),
            ('Em qual planeta nós vivemos?', 'Nós vivemos no planeta Terra.'),
            ('A que temperatura a água ferve?', 'A água ferve a 100 °C ao nível do mar.'),
            ('A que temperatura a água congela?', 'A água congela a 0 °C.'),
            ('Quantas letras tem o alfabeto?', 'O alfabeto português tem 26 letras.'),
        ),
        'echo_users': ('Diga {w}', 'Diga "{w}"', 'Repita comigo: {w}', 'Repita {w}', 'Pode dizer {w}?',
                       'Por favor, diga {w}'),
        'echo_replies': ('{w}', '{w}.', 'Claro! {w}', '{w}!'),
        'words': ('olá', 'banana', 'maçã', 'casa', 'música', 'oceano', 'sol', 'computador', 'amigo', 'pizza',
                  'robô', 'café', 'jardim', 'rio', 'janela', 'arco-íris'),
    },
}

# Peso de cada categoria nas conversas de 1 turno (matemática/capitais/fatos aparecem mais: são o que o
# modelo consegue de fato aprender a acertar; o resto ensina o "jeito" de conversar).
_CHAT_PESOS = {
    'greet': 8, 'morning': 2, 'afternoon': 2, 'evening': 2, 'night': 2, 'howru': 6, 'fine': 3, 'sad': 3, 'bored': 2,
    'name': 5, 'human_q': 2, 'ai_q': 3, 'creator': 2, 'can_do': 5, 'no_realtime': 4, 'browse': 2, 'thanks': 6,
    'bye': 5, 'compliment': 3, 'wrong': 3, 'rude': 2, 'help': 4, 'present': 2, 'joke': 4, 'feelings': 2,
    'favorite': 2, 'age': 1, 'where': 1, 'language': 2, 'ack': 3, 'gibberish': 2, 'unknown': 2, 'reasoning_q': 1,
    'math': 12, 'capital': 8, 'facts': 6, 'echo': 3, 'name_intro': 3, 'recall': 1,
}

# Conversas de vários turnos (cada item = sequência de categorias). 'recall' depois de 'name_intro'
# vira "lembrar o nome que o usuário acabou de dizer" (cópia de contexto).
_CHAT_FLUXOS = (
    ('greet', 'howru'), ('greet', 'name'), ('greet', 'can_do'), ('howru', 'fine'), ('greet', 'howru', 'fine'),
    ('greet', 'joke', 'thanks'), ('name_intro', 'recall'), ('greet', 'name_intro', 'recall'),
    ('can_do', 'math'), ('math', 'math'), ('math', 'thanks', 'bye'), ('sad', 'thanks'), ('joke', 'joke'),
    ('joke', 'compliment'), ('help', 'math'), ('help', 'capital'), ('capital', 'capital'),
    ('capital', 'thanks'), ('thanks', 'bye'), ('name', 'thanks'), ('morning', 'howru'), ('evening', 'howru'),
    ('bored', 'joke'), ('ai_q', 'can_do'), ('greet', 'name', 'can_do', 'thanks', 'bye'),
    ('facts', 'facts', 'thanks'), ('name_intro', 'howru', 'fine'), ('wrong', 'thanks'), ('reasoning_q', 'math'),
)


def _variar(rng, s: str) -> str:
    """Variações de digitação (minúsculas, sem pontuação final) — usuário real não escreve sempre 'Hello!'."""
    r = rng.random()
    if r < 0.12:
        return s.lower()
    if r < 0.20:
        return s.rstrip('?!.') or s
    return s


def _turno_chat(rng, lang: str, cat: str, ctx: dict):
    """Um par (mensagem do usuário, resposta) da categoria `cat`. `ctx` guarda o nome dito pelo usuário."""
    T = _CHAT_TXT[lang]
    if cat == 'math':
        op = rng.choice(('+', '-', 'x'))
        if op == '+':
            a, b = rng.randint(1, 50), rng.randint(1, 50)
            r = a + b
        elif op == '-':
            a = rng.randint(2, 60)
            b = rng.randint(1, a - 1)
            r = a - b
        else:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            r = a * b
        e = f'{a} {op} {b}'
        e_user = rng.choice(T['math_ops'][op]).format(a=a, b=b)
        return (rng.choice(T['math_frames']).format(e=e_user), rng.choice(T['math_replies']).format(e=e, r=r))
    if cat == 'capital':
        c, k = rng.choice(T['capitals'])
        return (rng.choice(T['cap_users']).format(c=c), rng.choice(T['cap_replies']).format(c=c, k=k))
    if cat == 'facts':
        return rng.choice(T['facts'])
    if cat == 'echo':
        w = rng.choice(T['words'])
        return (rng.choice(T['echo_users']).format(w=w), rng.choice(T['echo_replies']).format(w=w))
    if cat == 'name_intro':
        n = rng.choice(_CHAT_NOMES)
        ctx['nome'] = n
        return (rng.choice(T['name_intro'][0]).format(n=n), rng.choice(T['name_intro'][1]).format(n=n))
    if cat == 'recall':
        n = ctx.get('nome')
        if n:
            return (rng.choice(T['recall'][0]), rng.choice(T['recall_known']).format(n=n))
        return (rng.choice(T['recall'][0]), rng.choice(T['recall'][1]))
    users, replies = T['cats'][cat]
    return (rng.choice(users), rng.choice(replies))


def _gerar_chat_sintetico(n: int, lang: str = 'both', seed: int = 4321) -> str:
    """Gera n conversas sintéticas (~70% de 1 turno, ~30% de 2-5 turnos) em en, pt ou ambos.
    Cobre saudações, humor, identidade, limites (sem internet / sem hora), agradecimento, despedida,
    matemática direta, capitais, fatos simples e memória do nome dentro da conversa.
    Usa um Random local: não mexe no estado do random global."""
    rng = random.Random(seed)
    langs = ('en', 'pt') if lang == 'both' else (lang,)
    cats = list(_CHAT_PESOS)
    pesos = [_CHAT_PESOS[c] for c in cats]
    out = []
    for _ in range(n):
        L = rng.choice(langs)
        ctx = {}
        fluxo = rng.choice(_CHAT_FLUXOS) if rng.random() < 0.30 else (rng.choices(cats, pesos)[0],)
        turnos = []
        for cat in fluxo:
            u, a = _turno_chat(rng, L, cat, ctx)
            if cat not in ('echo', 'name_intro'):
                u = _variar(rng, u)
            turnos.append((u, a))
        out.append(_formatar_chat(turnos))
    return ''.join(out)


# ═══════════════════════════════════════════════════════════════════════
# CARREGAMENTO DO CORPUS (texto puro + CoT + chat)
# ═══════════════════════════════════════════════════════════════════════
_COL_Q = ('question', 'prompt', 'instruction', 'query', 'pergunta', 'input')
_COL_R = ('reasoning', 'cot', 'chain_of_thought', 'thought', 'thoughts', 'thinking', 'rationale', 'raciocinio', 'raciocínio', 'pensamento')
_COL_A = ('answer', 'final_answer', 'output', 'response', 'resposta')


def _detectar_cot(nomes):
    """Recebe nomes de colunas/campos. Se houver pergunta + raciocínio + resposta,
    devolve (nome_q, nome_r, nome_a) com os nomes ORIGINAIS; senão None.
    Exige a coluna de raciocínio de propósito: QA sem raciocínio ensinaria o
    modelo a pular o pensamento."""
    low = {str(n).strip().lower(): n for n in nomes}

    def pick(cands):
        for c in cands:
            if c in low:
                return low[c]
        return None
    q, r, a = pick(_COL_Q), pick(_COL_R), pick(_COL_A)
    return (q, r, a) if (q is not None and r is not None and a is not None) else None


def _como_texto(v) -> str:
    if isinstance(v, (list, tuple)):
        return ' '.join(str(x) for x in v)
    return '' if v is None else str(v)


_TAG_RACIOC = r'(?:think|thinking|thought|reasoning)'
_RE_RACIOC = _re.compile(
    r'^\s*(?:<' + _TAG_RACIOC + r'>)?(.*?)</' + _TAG_RACIOC + r'>\s*(.*?)\s*$', _re.S | _re.I)
_RE_ANSWER_TAG = _re.compile(r'^\s*<answer>(.*?)</answer>\s*$', _re.S | _re.I)


def _separar_raciocinio(out: str):
    """Aceita '<thinking>X</thinking><answer>Y</answer>', '<think>X</think>Y' ou 'X</think>Y'.
    Devolve (raciocínio, resposta) ou None se não houver raciocínio + resposta."""
    m = _RE_RACIOC.match(out or '')
    if not m:
        return None
    r, a = m.group(1).strip(), m.group(2).strip()
    ma = _RE_ANSWER_TAG.match(a)
    if ma:
        a = ma.group(1).strip()
    if len(a) >= 2 and a[0] == a[-1] == '"':
        a = a[1:-1].strip()
    return (r, a) if r and a else None


def _detectar_alpaca(nomes):
    """(instruction, input|None, output) com os nomes ORIGINAIS, ou None."""
    low = {str(n).strip().lower(): n for n in nomes}
    if 'instruction' in low and 'output' in low:
        return (low['instruction'], low.get('input'), low['output'])
    return None


def _alpaca_para_cot(instr: str, inp: str, out: str):
    """Alpaca com raciocínio dentro do output -> bloco CoT (ou None se não houver raciocínio:
    descartar de propósito, pra não ensinar o modelo a pular o pensamento).
    Se o input vier como 'User: ...', a instruction é só um rótulo da tarefa e a pergunta
    real é o input (é isso que o usuário digita na hora de perguntar)."""
    sep = _separar_raciocinio(out)
    if sep is None:
        return None
    inp = (inp or '').strip()
    instr = (instr or '').strip()
    if inp[:5].lower() == 'user:':
        q = inp[5:].strip()
    else:
        q = instr + (('\n' + inp) if inp else '')
    if not q.strip():
        return None
    return _formatar_cot(q, sep[0], sep[1])


def _detectar_gsm8k(nomes):
    """(question, answer) se o formato for GSM8K-like: answer = raciocínio + '#### resposta'."""
    low = {str(n).strip().lower(): n for n in nomes}
    if 'question' in low and 'answer' in low:
        return (low['question'], low['answer'])
    return None


def _gsm8k_para_cot(q: str, ans: str):
    if '####' not in ans:
        return None
    r, a = ans.rsplit('####', 1)
    r = _re.sub(r'<<[^>]*>>', '', r).strip()   # tira as anotações de calculadora <<48/2=24>>
    a = a.strip()
    if not q.strip() or not r or not a:
        return None
    return _formatar_cot(q, r, a)


def _registro_para_bloco(obj: dict):
    """Um registro (linha de csv/jsonl/parquet) -> (bloco, is_cot) ou None."""
    campos = _detectar_cot(obj.keys())
    if campos is not None:
        q, r, a = (_como_texto(obj.get(c)) for c in campos)
        if not q.strip() or not a.strip():
            return None
        return (_formatar_cot(q, r, a), True)
    alp = _detectar_alpaca(obj.keys())
    if alp is not None:
        ci, cn, co = alp
        bloco = _alpaca_para_cot(_como_texto(obj.get(ci)),
                                 _como_texto(obj.get(cn)) if cn is not None else '',
                                 _como_texto(obj.get(co)))
        if bloco is not None:
            return (bloco, True)
    gsm = _detectar_gsm8k(obj.keys())
    if gsm is not None:
        bloco = _gsm8k_para_cot(_como_texto(obj.get(gsm[0])), _como_texto(obj.get(gsm[1])))
        if bloco is not None:
            return (bloco, True)
    if isinstance(obj.get('text'), str) and obj['text'].strip():
        return (obj['text'], False)
    return None


def _juntar_registros(registros, nome_arquivo: str):
    """Consome dicts, respeita CSV_MAX_CHARS_POR_ARQUIVO. Retorna (texto, is_cot)."""
    blocos = []
    chars_cot = chars_txt = 0
    is_cot = False
    for obj in registros:
        if not isinstance(obj, dict):
            continue
        res = _registro_para_bloco(obj)
        if res is None:
            continue
        bloco, cot = res
        is_cot = is_cot or cot
        blocos.append(bloco if cot else bloco + '\n')
        if cot:
            chars_cot += len(bloco)
        else:
            chars_txt += len(bloco)
        if chars_cot >= COT_MAX_CHARS_POR_ARQUIVO or chars_txt >= CSV_MAX_CHARS_POR_ARQUIVO:
            break
    return (''.join(blocos), is_cot)


def _extrair_texto_csv(caminho_completo):
    """Retorna (texto, is_cot)."""
    texto = []
    total_chars = 0
    try:
        csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    except Exception:
        pass
    try:
        with open(caminho_completo, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            amostra = [f.readline() for _ in range(CSV_SAMPLE_LINHAS)]
            f.seek(0)
            sniffer = csv.Sniffer()
            try:
                delim = sniffer.sniff(''.join(amostra[:20]) or ',', delimiters=',;\t|').delimiter
            except Exception:
                delim = ','

            # só o delimitador vem do sniffer: ele costuma errar doublequote e quebra campos com ""
            class dialect(csv.excel):
                delimiter = delim
            reader = csv.reader(f, dialect)
            header = next(reader, None)
            if header is None:
                return ('', False)
            if (_detectar_cot(header) is not None or _detectar_alpaca(header) is not None
                    or _detectar_gsm8k(header) is not None):
                regs = (dict(zip(header, row)) for row in reader)
                bloco, is_cot = _juntar_registros(regs, caminho_completo)
                if bloco:
                    return (bloco, is_cot)
                # nada aproveitável como CoT (ex.: Alpaca sem raciocínio): cai pro texto puro
                f.seek(0)
                reader = csv.reader(f, dialect)
                header = next(reader, None)
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
        return ('', False)
    return ('\n'.join(texto), False)


def _extrair_texto_jsonl(caminho_completo):
    """Retorna (texto, is_cot). Cada linha é um objeto JSON: pergunta+raciocínio+resposta,
    Alpaca com raciocínio no output (vira CoT), ou um campo 'text' (vira texto puro)."""
    def registros():
        with open(caminho_completo, 'r', encoding='utf-8', errors='ignore') as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    yield _json.loads(linha)
                except Exception:
                    continue
    try:
        return _juntar_registros(registros(), caminho_completo)
    except Exception as e:
        print(f'Erro ao ler JSONL {os.path.basename(caminho_completo)}: {e}')
        return ('', False)


def _extrair_texto_parquet(caminho_completo):
    """Retorna (texto, is_cot). Mesmas regras do jsonl; lê em lotes (não carrega tudo na RAM)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(f'⚠️  {os.path.basename(caminho_completo)}: instale o pyarrow pra ler .parquet (pip install pyarrow)')
        return ('', False)

    def registros():
        pf = pq.ParquetFile(caminho_completo)
        for lote in pf.iter_batches(batch_size=1000):
            for obj in lote.to_pylist():
                yield obj
    try:
        return _juntar_registros(registros(), caminho_completo)
    except Exception as e:
        print(f'Erro ao ler PARQUET {os.path.basename(caminho_completo)}: {e}')
        return ('', False)


def carregar_corpus_categorizado(diretorios_raiz, categorias: dict):
    """Devolve {categoria: texto} + a chave extra 'cot' com tudo que já veio no
    formato CoT (jsonl/csv com pergunta+raciocínio+resposta, ou .txt com os tokens)."""
    if isinstance(diretorios_raiz, str):
        diretorios_raiz = [diretorios_raiz]
    partes = {nome: [] for nome in categorias}
    partes['cot'] = []
    partes['chat'] = []
    ext_para_categoria = {}
    for nome, exts in categorias.items():
        for ext in exts:
            ext_para_categoria[ext] = nome
    candidatos = []
    for diretorio_raiz in diretorios_raiz:
        if not os.path.isdir(diretorio_raiz):
            continue
        for pasta_atual, subpastas, arquivos in os.walk(diretorio_raiz):
            subpastas[:] = [d for d in subpastas if d not in IGNORE_DIRS]
            for nome_arquivo in arquivos:
                lower = nome_arquivo.lower()
                if lower in IGNORE_FILES:
                    continue
                ext_match = next((e for e in ext_para_categoria if lower.endswith(e)), None)
                if ext_match is None:
                    continue
                caminho = os.path.join(pasta_atual, nome_arquivo)
                try:
                    tam = os.path.getsize(caminho)
                except OSError:
                    tam = 0
                candidatos.append((tam, caminho, nome_arquivo, ext_match))
    candidatos.sort()   # menores primeiro: datasets curados pequenos (CoT, identidade) nunca são cortados pelo limite total
    arquivo_count = 0
    total_chars = 0
    for _, caminho_completo, nome_arquivo, ext_match in candidatos:
        if arquivo_count >= CORPUS_MAX_FILES or total_chars >= CORPUS_MAX_CHARS_TOTAL:
            print(f'⚠️  Limite de corpus atingido ({arquivo_count} arquivos, {total_chars} chars) — ignorando o resto (ex.: {nome_arquivo})')
            break
        lower = nome_arquivo.lower()
        categoria = ext_para_categoria[ext_match]
        is_chat = False
        try:
            if lower.endswith('.csv'):
                trecho, is_cot = _extrair_texto_csv(caminho_completo)
            elif lower.endswith('.jsonl'):
                trecho, is_cot = _extrair_texto_jsonl(caminho_completo)
            elif lower.endswith('.parquet'):
                trecho, is_cot = _extrair_texto_parquet(caminho_completo)
            else:
                with open(caminho_completo, 'r', encoding='utf-8', errors='ignore') as f:
                    trecho = f.read(TXT_MAX_CHARS_POR_ARQUIVO + 1)
                if len(trecho) > TXT_MAX_CHARS_POR_ARQUIVO:
                    trecho = trecho[:TXT_MAX_CHARS_POR_ARQUIVO]
                    print(f'✂️  {nome_arquivo}: só os primeiros {TXT_MAX_CHARS_POR_ARQUIVO} chars foram lidos (TXT_MAX_CHARS_POR_ARQUIVO)')
                is_cot = THINK_TOKEN in trecho
                is_chat = (not is_cot) and CHAT_TOKEN in trecho
                if is_cot:
                    trecho = _normalizar_cot_txt(trecho)
            if not trecho:
                print(f'Vazio/ignorado: {nome_arquivo}')
                continue
            destino = 'cot' if is_cot else ('chat' if is_chat else categoria)
            partes[destino].append(trecho)
            total_chars += len(trecho)
            arquivo_count += 1
            print(f'Lido ({destino}): {nome_arquivo} ({len(trecho)} chars)')
        except Exception as e:
            print(f'Erro ao ler {nome_arquivo}: {e}')
    return {k: '\n'.join(v) for k, v in partes.items()}


try:
    diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
except NameError:
    diretorio_raiz = os.getcwd()
# Pastas extras de corpus (além da pasta do script): Kaggle, /data/corpus (storage persistente do
# HF Space, onde dá pra colocar datasets grandes sem estourar o repo) e CORPUS_DIRS (separadas por ':').
EXTRA_CORPUS_DIRS = [d for d in ['/kaggle/input', '/data/corpus', '/data'] + [x for x in os.environ.get('CORPUS_DIRS', '').split(os.pathsep) if x]
                     if os.path.isdir(d)]
_CATEGORIAS_CORPUS = {'general': ('.txt', '.csv', '.jsonl', '.parquet')}
_corpus_por_categoria = carregar_corpus_categorizado([diretorio_raiz] + EXTRA_CORPUS_DIRS, _CATEGORIAS_CORPUS)
CORPUS_GENERAL = _corpus_por_categoria['general']
CORPUS_COT_USER = _corpus_por_categoria['cot']
CORPUS_COT_SINTETICO = _gerar_cot_sintetico(COT_SYNTH_EXAMPLES, COT_SYNTH_LANG) if COT_SYNTH_EXAMPLES > 0 else ''
CORPUS_COT = CORPUS_COT_USER + CORPUS_COT_SINTETICO
CORPUS_CHAT_USER = _corpus_por_categoria['chat']
CORPUS_CHAT_SINTETICO = _gerar_chat_sintetico(CHAT_SYNTH_EXAMPLES, CHAT_SYNTH_LANG) if CHAT_SYNTH_EXAMPLES > 0 else ''
CORPUS_CHAT = CORPUS_CHAT_USER + CORPUS_CHAT_SINTETICO
CORPUS = CORPUS_GENERAL + '\n' + CORPUS_COT + '\n' + CORPUS_CHAT   # usado só pra treinar o tokenizer

# ═══════════════════════════════════════════════════════════════════════
# TOKENIZER — BPE byte-level + 3 tokens especiais do CoT
# ═══════════════════════════════════════════════════════════════════════
BPE_VOCAB_SIZE = int(os.environ.get('BPE_VOCAB_SIZE', 4096))  # tamanho do BPE, SEM contar os especiais
# (tok_emb agora é compartilhada com a head — weight tying — então vocab maior custa metade do que custava.
#  Se mudar isto, apague checkpoints/tokenizer.json pra retreinar os merges.)


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


_TOKENIZER_CACHE_PATH = os.path.join(CHECKPOINT_DIR, 'tokenizer.json')


def _salvar_cache_tokenizer(merges: dict):
    try:
        os.makedirs(os.path.dirname(_TOKENIZER_CACHE_PATH) or '.', exist_ok=True)
        with open(_TOKENIZER_CACHE_PATH, 'w', encoding='utf-8') as f:
            _json.dump({'vocab_size': BPE_VOCAB_SIZE, 'merges': _bpe_merges_to_json(merges)}, f)
    except Exception as e:
        console.print(f'[yellow]⚠️  Não deu pra salvar cache do tokenizer: {e}[/yellow]')


def _amostra_espalhada(texto: str, max_chars: int, n_fatias: int = 64) -> str:
    """Pega n_fatias pedaços igualmente espaçados (representa o corpus todo, não só o começo)."""
    if len(texto) <= max_chars:
        return texto
    tam = max(1, max_chars // n_fatias)
    passo = len(texto) // n_fatias
    return '\n'.join(texto[i * passo:i * passo + tam] for i in range(n_fatias))


def _bpe_load_or_train(corpus_text: str, vocab_size: int, cache_path: str) -> dict:
    """Carrega os merges do cache em disco ou treina de novo (BPE em Python puro
    não é instantâneo em corpus grande). Os tokens especiais do CoT são removidos
    do texto antes do treino pra não poluir os merges."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                saved = _json.load(f)
            if saved.get('vocab_size') == vocab_size:
                console.print(f'[bold cyan]🔤 Tokenizer BPE carregado do cache ({cache_path})[/bold cyan]')
                return _bpe_merges_from_json(saved['merges'])
        except Exception:
            pass
    amostra = _amostra_espalhada(corpus_text, BPE_TRAIN_MAX_CHARS)
    console.print(f'[bold cyan]🔤 Treinando tokenizer BPE (vocab_size={vocab_size}) numa amostra de '
                  f'{len(amostra):,} de {len(corpus_text):,} chars...[/bold cyan]')
    merges = _bpe_train(_SPECIAL_RE.sub('', amostra), vocab_size)
    _salvar_cache_tokenizer(merges)
    console.print(f'[bold cyan]🔤 Tokenizer BPE pronto: {256 + len(merges)} tokens ({len(merges)} merges) + {len(SPECIAL_TOKENS)} especiais[/bold cyan]')
    return merges


BPE_MERGES: dict = {}
BPE_VOCAB: dict = {}
VOCAB = 0
SPECIAL_BASE = 0
Q_ID = THINK_ID = ANSWER_ID = 0
CHAT_ID = COT_ID = USER_ID = ASSISTANT_ID = 0
_SPECIAL_IDS: dict = {}
EOS_ID = 0        # byte 0, protegido de merges em _bpe_train
NEWLINE_ID = 10   # byte 10 — atômico (nunca fundido)


def _set_tokenizer(merges: dict):
    """Fixa o tokenizer global. Ids: [0..255] bytes | [256..SPECIAL_BASE-1] merges | especiais no fim."""
    global BPE_MERGES, BPE_VOCAB, VOCAB, SPECIAL_BASE, Q_ID, THINK_ID, ANSWER_ID, _SPECIAL_IDS
    global CHAT_ID, COT_ID, USER_ID, ASSISTANT_ID
    BPE_MERGES = merges
    vocab = _bpe_vocab_from_merges(merges)
    SPECIAL_BASE = len(vocab)
    _SPECIAL_IDS = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        vocab[SPECIAL_BASE + i] = tok.encode('utf-8')
        _SPECIAL_IDS[tok] = SPECIAL_BASE + i
    BPE_VOCAB = vocab
    VOCAB = len(vocab)
    Q_ID, THINK_ID, ANSWER_ID = (_SPECIAL_IDS[Q_TOKEN], _SPECIAL_IDS[THINK_TOKEN], _SPECIAL_IDS[ANSWER_TOKEN])
    CHAT_ID, COT_ID = _SPECIAL_IDS[CHAT_TOKEN], _SPECIAL_IDS[COT_TOKEN]
    USER_ID, ASSISTANT_ID = _SPECIAL_IDS[USER_TOKEN], _SPECIAL_IDS[ASSISTANT_TOKEN]


def encode(s: str) -> list:
    out = []
    for parte in _SPECIAL_RE.split(s):
        if not parte:
            continue
        sid = _SPECIAL_IDS.get(parte)
        if sid is not None:
            out.append(sid)
        else:
            out.extend(_bpe_encode(parte, BPE_MERGES))
    return out


def decode(ids: list) -> str:
    return _bpe_decode(ids, BPE_VOCAB)


def _ids_to_tensor(ids: list) -> torch.Tensor:
    """Guarda os tokens em int16 (2 bytes/token, em vez dos 8 do int64) — pra um corpus de
    ~100MB isso é ~120MB em RAM em vez de ~480MB. Os batches são convertidos pra long na hora de usar."""
    code, tdtype = ('h', torch.int16) if VOCAB < 32000 else ('i', torch.int32)
    if not ids:
        return torch.zeros(0, dtype=tdtype)
    arr = array.array(code, ids)
    return torch.frombuffer(arr, dtype=tdtype).clone()


data_tensor = torch.zeros(0, dtype=torch.int16)   # texto puro
cot_tensor = torch.zeros(0, dtype=torch.int16)    # exemplos CoT (usuário + sintético)
chat_tensor = torch.zeros(0, dtype=torch.int16)   # conversas (usuário + sintético)


def _encode_para_tensor(texto: str, rotulo: str, chunk_chars: int = 1000000) -> torch.Tensor:
    """encode() em pedaços de ~1M de chars, com progresso. O BPE em Python puro gasta ~300 bytes de RAM
    por caractere, então codificar o corpus inteiro de uma vez estoura a memória e não mostra nada.
    Os cortes caem em '\\n' ou no EOS (\\x00), então só as bordas dos pedaços podem tokenizar diferente."""
    code, tdtype = ('h', torch.int16) if VOCAB < 32000 else ('i', torch.int32)
    arr = array.array(code)
    n = len(texto)
    if n == 0:
        return torch.zeros(0, dtype=tdtype)
    pos, t0, ultimo_log = 0, time.time(), time.time()
    while pos < n:
        fim = min(n, pos + chunk_chars)
        if fim < n:
            c = max(texto.rfind('\n', pos, fim), texto.rfind('\x00', pos, fim))
            if c <= pos:
                c = texto.find('\n', fim)
                c = n - 1 if c == -1 else c
            fim = c + 1
        arr.extend(encode(texto[pos:fim]))
        pos = fim
        agora = time.time()
        if agora - ultimo_log >= 15 or pos >= n:
            ultimo_log = agora
            dec = agora - t0
            eta = dec / pos * (n - pos) if pos else 0
            console.print(f'   🔢 encode {rotulo}: {pos / n:5.1%}  ({len(arr):,} tokens, {dec:.0f}s, ~{eta:.0f}s restantes)')
    return torch.frombuffer(arr, dtype=tdtype).clone()


_CORPUS_CACHE_PATH = os.path.join(CHECKPOINT_DIR, 'corpus_tokens.pt')


def _assinaturas_corpus() -> dict:
    """Uma assinatura por parte (texto / CoT / chat), todas dependentes do tokenizer + tokens especiais.
    Assim, mudar só o chat ou o CoT NÃO obriga a recodificar o corpus de texto (que é o enorme)."""
    base = hashlib.sha1(_json.dumps(_bpe_merges_to_json(BPE_MERGES)).encode())
    base.update('|'.join(SPECIAL_TOKENS).encode())
    sigs = {}
    for nome, parte in (('data', CORPUS_GENERAL), ('cot', CORPUS_COT), ('chat', CORPUS_CHAT)):
        h = base.copy()
        h.update(parte.encode('utf-8', 'ignore'))
        sigs[nome] = h.hexdigest()
    return sigs


def _rebuild_corpus_tensors():
    """Codifica o corpus (texto + CoT + chat). Cada parte fica em disco (corpus_tokens.pt) com a própria
    assinatura: nos próximos boots só as partes que mudaram são recodificadas (o encode leva minutos+).
    Cache no formato antigo (sem 'sigs'): recodifica tudo, a não ser que REUSE_LEGACY_TEXT_CACHE=1 — aí o
    tensor de TEXTO antigo é reaproveitado sem verificação (só use se o corpus de texto e o tokenizer não mudaram)."""
    global data_tensor, cot_tensor, chat_tensor
    sigs = _assinaturas_corpus()
    cache = {}
    if os.path.exists(_CORPUS_CACHE_PATH):
        try:
            cache = torch.load(_CORPUS_CACHE_PATH, map_location='cpu')
        except Exception as e:
            console.print(f'[yellow]Cache do corpus inválido ({e}); recodificando.[/yellow]')
            cache = {}
    cache_sigs = cache.get('sigs') if isinstance(cache.get('sigs'), dict) else {}
    legado_ok = (not cache_sigs and 'data' in cache and os.environ.get('REUSE_LEGACY_TEXT_CACHE', '0') == '1')
    if legado_ok:
        console.print('[bold yellow]♻️  REUSE_LEGACY_TEXT_CACHE=1: reaproveitando o texto tokenizado do cache antigo.[/bold yellow]')
    partes = (('data', CORPUS_GENERAL, 'texto'), ('cot', CORPUS_COT, 'CoT'), ('chat', CORPUS_CHAT, 'chat'))
    tensores, mudou = {}, False
    for nome, texto, rotulo in partes:
        if nome in cache and cache_sigs.get(nome) == sigs[nome]:
            tensores[nome] = cache[nome]
        elif nome == 'data' and legado_ok:
            tensores[nome] = cache['data']
            mudou = True   # regrava o cache já no formato novo
        else:
            console.print(f'[bold cyan]🔢 Codificando {rotulo}: {len(texto):,} chars[/bold cyan]')
            tensores[nome] = _encode_para_tensor(texto, rotulo)
            mudou = True
    data_tensor, cot_tensor, chat_tensor = tensores['data'], tensores['cot'], tensores['chat']
    resumo = f'texto={len(data_tensor):,} | CoT={len(cot_tensor):,} | chat={len(chat_tensor):,} tokens'
    if not mudou:
        console.print(f'[bold cyan]⚡ Corpus tokenizado carregado do cache: {resumo}[/bold cyan]')
        return
    try:
        os.makedirs(os.path.dirname(_CORPUS_CACHE_PATH) or '.', exist_ok=True)
        tmp = _CORPUS_CACHE_PATH + '.tmp'
        torch.save({'sigs': sigs, 'data': data_tensor, 'cot': cot_tensor, 'chat': chat_tensor}, tmp)
        os.replace(tmp, _CORPUS_CACHE_PATH)
        console.print(f'[bold cyan]💾 Corpus tokenizado salvo em cache: {resumo}[/bold cyan]')
    except Exception as e:
        console.print(f'[yellow]Não consegui salvar o cache do corpus: {e}[/yellow]')


_set_tokenizer(_bpe_load_or_train(CORPUS, BPE_VOCAB_SIZE, _TOKENIZER_CACHE_PATH))
_rebuild_corpus_tensors()

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


def _amp():
    """bf16 só em GPU Ampere+ (capability >= 8). Em T4/P100 (Kaggle/Colab) fica em fp32
    puro — fp16 exigiria GradScaler e não vale a complexidade aqui."""
    if device.type == 'cuda':
        try:
            if torch.cuda.get_device_capability()[0] >= 8:
                return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        except Exception:
            pass
    return contextlib.nullcontext()


# ═══════════════════════════════════════════════════════════════════════
# MODELO — GPT denso (sem MoE)
# ═══════════════════════════════════════════════════════════════════════
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
        # Sem buffer de máscara causal: F.scaled_dot_product_attention(is_causal=True) já cuida disso
        # (e usa kernel flash/mem-efficient). O buffer antigo custava 2048*2048*4B = 16MB POR CAMADA
        # dentro do checkpoint, e a atenção "ingênua" alocava a matriz T×T inteira em cada camada.

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
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=p)  # 1 token novo enxerga tudo
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
        ed = cfg['embed_dim']
        nh = cfg['n_heads']
        nl = cfg['n_layers']
        do = cfg['dropout']
        self.tok_emb = nn.Embedding(VOCAB, ed)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, ed)
        self.drop = nn.Dropout(do)
        self.blocks = nn.ModuleList([Block(ed, nh, do) for _ in range(nl)])
        self.ln_f = nn.LayerNorm(ed)
        self.head = nn.Linear(ed, VOCAB, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying: economiza VOCAB*embed_dim params
        self.apply(self._init)
        # init escalada nas projeções que escrevem no fluxo residual (estilo GPT-2) — treino mais estável
        for name, p in self.named_parameters():
            if name.endswith('attn.out.weight') or name.endswith('mlp.proj.weight'):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * nl))
        self._cfg = cfg

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

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
            x = x[:, -1:, :]   # na geração só o último token interessa — evita calcular a head no prompt inteiro
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return (logits, loss, new_kvs)

    @property
    def n_params(self):
        return sum((p.numel() for p in self.parameters()))  # parameters() já deduplica o peso amarrado


# ═══════════════════════════════════════════════════════════════════════
# CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════════
def _ckpt_path(filename: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, filename)


def _atomic_torch_save(obj, path: str) -> None:
    """Escreve em .tmp e renomeia — se o processo morrer no meio (ex: --max-minutes / OOM),
    o checkpoint anterior continua íntegro em vez de ficar truncado."""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(model: 'TinyAI', opt, state: dict, filename: str = CHECKPOINT_LAST) -> None:
    path = _ckpt_path(filename)
    payload = {
        'format': CKPT_FORMAT,
        'model_cfg': model._cfg,
        'model_state': model.state_dict(),
        'bpe_merges': _bpe_merges_to_json(BPE_MERGES),
        'vocab_size': VOCAB,
        'context_len': CONTEXT_LEN,
        'opt_state': opt.state_dict() if (opt is not None and SAVE_OPT_STATE) else None,
        'gen': state.get('gen', 0),
    }
    _atomic_torch_save(payload, path)


def export_weights(model: 'TinyAI', filename: str = WEIGHTS_EXPORT) -> str:
    """Só pesos em fp16 + tokenizer + config (~2 bytes/param) — é o que vale distribuir/servir.
    Sem estado do otimizador. 'head.weight' fica de fora porque é o mesmo tensor de tok_emb."""
    path = _ckpt_path(filename)
    sd = {k: (v.detach().half() if v.is_floating_point() else v.detach())
          for k, v in model.state_dict().items() if k != 'head.weight'}
    payload = {'format': CKPT_FORMAT, 'model_cfg': model._cfg, 'model_state': sd,
               'bpe_merges': _bpe_merges_to_json(BPE_MERGES), 'vocab_size': VOCAB,
               'context_len': CONTEXT_LEN, 'opt_state': None, 'gen': 0}
    _atomic_torch_save(payload, path)
    return path


def load_checkpoint(filename: str = CHECKPOINT_LAST):
    path = _ckpt_path(filename)
    if not os.path.exists(path):
        return None
    # map_location='cpu': evita jogar um checkpoint antigo gigante na VRAM só pra descobrir que é incompatível.
    return torch.load(path, map_location='cpu', weights_only=False)


def _transplant_state_dict(saved_sd: dict, model: 'TinyAI', skip_prefixes: tuple = ()) -> None:
    dst_sd = model.state_dict()
    for key in dst_sd:
        if key not in saved_sd:
            continue
        if any(key.startswith(p) for p in skip_prefixes):
            continue
        s, d = (saved_sd[key], dst_sd[key])
        if s.shape == d.shape:
            dst_sd[key] = s.clone().to(d.dtype)
        else:
            slices = tuple((slice(0, min(a, b)) for a, b in zip(s.shape, d.shape)))
            dst_sd[key][slices] = s[slices].clone().to(d.dtype)
    model.load_state_dict(dst_sd)


def _novo_otimizador(model):
    return torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.0001)


def novo_estado_inicial():
    return dict(gen=0, n_params=0, config=INIT_CONFIG.copy())


def restore_from_checkpoint(ckpt: dict, state: dict):
    """Retorna (model, opt, vocab_mismatch)."""
    if 'bpe_merges' in ckpt:
        # O tokenizer é parte do checkpoint, não do ambiente: usa os merges exatos de quando foi salvo.
        ckpt_merges = _bpe_merges_from_json(ckpt['bpe_merges'])
        if ckpt_merges != BPE_MERGES:
            _set_tokenizer(ckpt_merges)
            _salvar_cache_tokenizer(ckpt_merges)
            _rebuild_corpus_tensors()
    model = TinyAI(ckpt['model_cfg']).to(device)
    saved_sd = dict(ckpt['model_state'])
    if 'head.weight' not in saved_sd:
        saved_sd['head.weight'] = saved_sd['tok_emb.weight']   # exportado sem a head (peso amarrado)
    ckpt_vocab = saved_sd['tok_emb.weight'].shape[0]
    ckpt_context_len = ckpt.get('context_len', saved_sd['pos_emb.weight'].shape[0])
    vocab_changed = ckpt_vocab != VOCAB
    context_changed = ckpt_context_len != CONTEXT_LEN
    vocab_mismatch = vocab_changed or context_changed
    if vocab_mismatch:
        avisos = []
        if vocab_changed:
            avisos.append(f'vocab: checkpoint={ckpt_vocab} tokens, atual={VOCAB} tokens ({VOCAB - ckpt_vocab:+d})')
        if context_changed:
            avisos.append(f'context_len: checkpoint={ckpt_context_len}, atual={CONTEXT_LEN}')
        console.print('[bold yellow]⚠️  ' + ' | '.join(avisos) + '[/bold yellow]\n   → Transplantando pesos compatíveis (posições/tokens novos ficam com init aleatório).\n   → Pré-treino de recuperação será executado automaticamente.')
        _transplant_state_dict(saved_sd, model)
        opt = _novo_otimizador(model)
    else:
        model.load_state_dict(saved_sd)
        opt = _novo_otimizador(model)
        opt_state = ckpt.get('opt_state')
        if opt_state is not None:
            try:
                opt.load_state_dict(opt_state)
            except Exception:
                pass
    state['gen'] = ckpt.get('gen', 0)
    state['config'] = ckpt['model_cfg']
    state['n_params'] = model.n_params
    return (model, opt, vocab_mismatch)


def _arquivar_checkpoint_legado():
    path = _ckpt_path(CHECKPOINT_LAST)
    destino = path + '.legacy'
    try:
        os.replace(path, destino)
        console.print(f'[bold yellow]📦 Checkpoint antigo (MoE / formato v1) é incompatível com a arquitetura nova.\n'
                      f'   Movido para {destino} — pode apagar pra liberar ~1GB. Começando um modelo novo.[/bold yellow]')
    except OSError as e:
        console.print(f'[bold yellow]⚠️  Checkpoint antigo incompatível e não consegui renomear ({e}). Ignorando.[/bold yellow]')


def _carregar_ou_criar_modelo(state):
    """Retorna (model, opt, modo) com modo in {'novo', 'continuacao', 'recuperacao'}."""
    ckpt = load_checkpoint(CHECKPOINT_LAST)
    if ckpt is not None and ckpt.get('format') != CKPT_FORMAT:
        del ckpt
        _arquivar_checkpoint_legado()
        ckpt = None
    if ckpt is not None:
        console.print('[bold yellow]♻️  Checkpoint encontrado! Carregando...[/bold yellow]')
        model, opt, vocab_mismatch = restore_from_checkpoint(ckpt, state)
        return (model, opt, 'recuperacao' if vocab_mismatch else 'continuacao')
    console.print('[bold green]🆕 Nenhum checkpoint encontrado. Criando modelo novo.[/bold green]')
    model = TinyAI(INIT_CONFIG).to(device)
    state['n_params'] = model.n_params
    return (model, _novo_otimizador(model), 'novo')


# ═══════════════════════════════════════════════════════════════════════
# BATCHES
# ═══════════════════════════════════════════════════════════════════════
def _janelas(tensor: torch.Tensor, n: int):
    """n janelas aleatórias (x, y) de CONTEXT_LEN tokens."""
    max_i = len(tensor) - CONTEXT_LEN - 1
    if n <= 0 or max_i <= 0:
        return ([], [])
    ix = torch.randint(0, max_i, (n,)).tolist()
    return ([tensor[i:i + CONTEXT_LEN] for i in ix], [tensor[i + 1:i + CONTEXT_LEN + 1] for i in ix])


def get_batch(bs: int = BATCH_SIZE):
    """Retorna (x, y, n_cot, n_chat). Ordem das linhas: [CoT | chat | texto puro]. Cada linha sorteia a
    fonte com probabilidade COT_BATCH_FRACTION / CHAT_BATCH_FRACTION / (o resto = texto). Fonte vazia
    é ignorada (as outras ficam com o batch todo)."""
    tem = {'cot': len(cot_tensor) > CONTEXT_LEN + 1, 'chat': len(chat_tensor) > CONTEXT_LEN + 1,
           'texto': len(data_tensor) > CONTEXT_LEN + 1}
    if not any(tem.values()):
        return (None, None, 0, 0)
    pesos = {'cot': COT_BATCH_FRACTION if tem['cot'] else 0.0,
             'chat': CHAT_BATCH_FRACTION if tem['chat'] else 0.0,
             'texto': max(1.0 - COT_BATCH_FRACTION - CHAT_BATCH_FRACTION, 0.01) if tem['texto'] else 0.0}
    if sum(pesos.values()) <= 0:   # frações zeradas e só sobrou a fonte de chat/CoT
        pesos = {k: (1.0 if v else 0.0) for k, v in tem.items()}
    fontes = random.choices(('cot', 'chat', 'texto'), weights=(pesos['cot'], pesos['chat'], pesos['texto']), k=bs)
    n_cot, n_chat = fontes.count('cot'), fontes.count('chat')
    xs_c, ys_c = _janelas(cot_tensor, n_cot)
    xs_h, ys_h = _janelas(chat_tensor, n_chat)
    xs_t, ys_t = _janelas(data_tensor, bs - n_cot - n_chat)
    xs, ys = xs_c + xs_h + xs_t, ys_c + ys_h + ys_t
    if not xs:
        return (None, None, 0, 0)
    return (torch.stack(xs).long().to(device), torch.stack(ys).long().to(device), n_cot, n_chat)


def pretrain(model, steps, lr, status_callback=None, opt=None):
    """status_callback(step, steps, loss_texto, loss_cot, loss_chat) -> True pra parar.
    loss_texto = loss só das linhas de texto puro (base do critério de parada);
    loss_cot / loss_chat = loss das linhas CoT / chat do batch (None se não houve)."""
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    for step in range(steps):
        x, y, n_cot, n_chat = get_batch()
        if x is None:
            break
        f = min(1.0, (step + 1) / max(1, PRETRAIN_WARMUP_STEPS))
        for pg in opt.param_groups:
            pg['lr'] = lr * f
        with _amp():
            logits, _, _ = model(x)
        B = x.size(0)
        por_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction='none').view(B, -1)
        loss = por_token.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            n_esp = n_cot + n_chat
            loss_texto = por_token[n_esp:].mean().item() if n_esp < B else loss.item()
            loss_cot = por_token[:n_cot].mean().item() if n_cot > 0 else None
            loss_chat = por_token[n_cot:n_esp].mean().item() if n_chat > 0 else None
        if status_callback and status_callback(step, steps, loss_texto, loss_cot, loss_chat):
            break
    return model


# ═══════════════════════════════════════════════════════════════════════
# GERAÇÃO (texto livre e Chain-of-Thought)
# ═══════════════════════════════════════════════════════════════════════
def _truncate_kv_cache(past_kv, max_len: int):
    """Corta o cache K/V pra caber em CONTEXT_LEN (pos_emb só tem CONTEXT_LEN posições)."""
    if past_kv is None:
        return None
    if max_len <= 0:
        return None
    out = []
    for k, v in past_kv:
        if k.shape[2] > max_len:
            k = k[:, :, -max_len:, :].contiguous()
            v = v[:, :, -max_len:, :].contiguous()
        out.append((k, v))
    return out


def _feed(model, ids: list, past_kv):
    tok = torch.tensor([ids], dtype=torch.long, device=device)
    past_kv = _truncate_kv_cache(past_kv, CONTEXT_LEN - len(ids))
    logits, _, past_kv = model(tok, past_kv=past_kv, use_cache=True, only_last=True)
    return (logits[:, -1, :], past_kv)


def _sample(logits, temperature: float, top_k: int = GEN_TOP_K, ban: tuple = ()) -> int:
    logits = logits.float() / max(temperature, 0.1)
    if ban:
        logits[:, list(ban)] = float('-inf')
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_text(model, prompt: str, max_new: int = 200, temperature: float = 0.8) -> str:
    """Geração livre (continuação de texto), sem estrutura de CoT."""
    was_training = model.training
    model.eval()
    try:
        tokens = encode(prompt) or [NEWLINE_ID]
        logits, past = _feed(model, tokens[-(CONTEXT_LEN - 1):], None)
        out = []
        for _ in range(max_new):
            tid = _sample(logits, temperature)
            if tid == EOS_ID:
                break
            out.append(tid)
            logits, past = _feed(model, [tid], past)
        return decode(out)
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def generate_cot(model, prompt: str, max_think: int = COT_MAX_THINK_TOKENS, max_answer: int = 200,
                 temperature: float = 0.8):
    """Chain-of-Thought em duas fases. Retorna (raciocinio, resposta).
      1) <|cot|><|q|>pergunta\\n<|think|>  -> o modelo gera o raciocínio até emitir <|answer|>
         (se estourar max_think sem emitir, o <|answer|> é FORÇADO pra ele sempre responder);
      2) depois de <|answer|>       -> gera a resposta até o EOS.
    Tokens estruturais são banidos onde não fazem sentido, então o formato nunca quebra."""
    was_training = model.training
    model.eval()
    try:
        # codifica pergunta+'\n' num pedaço só (igual ao treino: um '?\n' pode ser um único token BPE)
        corpo = encode(_limpar_entrada(prompt).strip() + '\n')[-(CONTEXT_LEN - 8):]
        ids = [COT_ID, Q_ID] + corpo + [THINK_ID]
        logits, past = _feed(model, ids, None)
        pensamento = []
        for _ in range(max_think):
            tid = _sample(logits, temperature, ban=(Q_ID, THINK_ID, EOS_ID, CHAT_ID, COT_ID, USER_ID, ASSISTANT_ID))
            if tid == ANSWER_ID:
                break
            pensamento.append(tid)
            logits, past = _feed(model, [tid], past)
        logits, past = _feed(model, [ANSWER_ID], past)   # emitido pelo modelo OU forçado (teto de pensamento)
        resposta = []
        for _ in range(max_answer):
            tid = _sample(logits, temperature, ban=(Q_ID, THINK_ID, ANSWER_ID, CHAT_ID, COT_ID, USER_ID, ASSISTANT_ID))
            if tid == EOS_ID:
                break
            resposta.append(tid)
            logits, past = _feed(model, [tid], past)
        return (decode(pensamento).strip(), decode(resposta).strip())
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def generate_chat(model, mensagem: str, historico=None, max_new: int = 200, temperature: float = 0.8) -> str:
    """Modo conversa: <|chat|> + turnos anteriores + <|user|>mensagem\n<|assistant|> -> o modelo responde até
    emitir EOS (ou começar um novo turno). `historico` = [(user, assistant), ...]. Se o contexto estourar,
    descarta os turnos mais antigos INTEIROS (o <|chat|> do começo nunca é cortado)."""
    was_training = model.training
    model.eval()
    try:
        mensagem = _limpar_entrada(mensagem)
        historico = [(_limpar_entrada(u), _limpar_entrada(a)) for u, a in (historico or [])]
        limite = CONTEXT_LEN - 8
        while True:
            ids = encode(_montar_prompt_chat(historico, mensagem))
            if len(ids) <= limite or not historico:
                break
            historico = historico[1:]
        if len(ids) > limite:   # só a mensagem já é longa demais: mantém <|chat|> + o final
            ids = [CHAT_ID] + ids[-(limite - 1):]
        logits, past = _feed(model, ids, None)
        ban = (Q_ID, THINK_ID, ANSWER_ID, CHAT_ID, COT_ID)
        out = []
        for _ in range(max_new):
            tid = _sample(logits, temperature, ban=ban)
            if tid in (EOS_ID, USER_ID, ASSISTANT_ID):   # fim da resposta
                break
            out.append(tid)
            logits, past = _feed(model, [tid], past)
        return decode(out).strip()
    finally:
        if was_training:
            model.train()


def _print_chat_sample(model):
    """Amostra rápida de chat durante o treino."""
    try:
        perguntas = {'en': ('Hello!', 'How are you?', 'What is your name?', 'What can you do?', 'Tell me a joke',
                            'What is 7 + 5?', 'Thank you!'),
                     'pt': ('Olá!', 'Como você está?', 'Qual é o seu nome?', 'O que você pode fazer?',
                            'Conte uma piada', 'Quanto é 7 + 5?', 'Obrigado!')}
        q = random.choice(perguntas['pt' if CHAT_SYNTH_LANG == 'pt' else 'en'])
        resp = generate_chat(model, q, max_new=60, temperature=0.3)
        console.print(f'   [dim]💬 {escape(q)}\n      resposta: {escape(resp)}[/dim]')
    except Exception as e:
        console.print(f'   [dim]💬 (amostra de chat falhou: {escape(str(e))})[/dim]')


def _print_cot_sample(model):
    """Amostra rápida de CoT durante o treino, pra dar pra ver o modelo aprendendo a raciocinar."""
    try:
        a, b = random.randint(10, 99), random.randint(10, 99)
        q = _COT_TXT[COT_SYNTH_LANG]['soma_q'].format(a=a, b=b)
        pensou, resp = generate_cot(model, q, max_think=160, max_answer=16, temperature=0.3)
        console.print(f'   [dim]🧠 {escape(q)}\n      pensou: {escape(pensou)}\n      resposta: {escape(resp)}  (esperado: {a + b})[/dim]')
    except Exception as e:
        console.print(f'   [dim]🧠 (amostra de CoT falhou: {escape(str(e))})[/dim]')


# ═══════════════════════════════════════════════════════════════════════
# TREINO OFFLINE (python ia_server.py)
# ═══════════════════════════════════════════════════════════════════════
def main():
    """Rotina de `python ia_server.py`: treina, salva o checkpoint e encerra."""
    console.print('\n[bold blue]══ 🧬 TinyAI — Treino (denso + Chain-of-Thought) ══[/bold blue]')
    console.print(f'   {dev_str}')
    console.print(f'   vocab={VOCAB} tokens (BPE + {len(SPECIAL_TOKENS)} especiais)  |  context={CONTEXT_LEN}  |  preset={MODEL_PRESET}')
    console.print(f'   texto={len(data_tensor)} tokens  |  CoT={len(cot_tensor)} tokens (~{COT_BATCH_FRACTION:.0%} das linhas do batch)'
                  f'  |  chat={len(chat_tensor)} tokens (~{CHAT_BATCH_FRACTION:.0%})\n')
    if (len(data_tensor) <= CONTEXT_LEN + 1 and len(cot_tensor) <= CONTEXT_LEN + 1
            and len(chat_tensor) <= CONTEXT_LEN + 1):
        console.print('[bold red]❌ Corpus curto demais pra treinar (coloque arquivos .txt/.csv/.jsonl/.parquet na pasta).[/bold red]\n')
        return
    state = novo_estado_inicial()
    model, opt, modo = _carregar_ou_criar_modelo(state)
    if modo == 'recuperacao':
        steps, lr, label = 2500, PRETRAIN_LR, 'recuperação'
    elif modo == 'continuacao':
        steps, lr, label = PRETRAIN_STEPS, PRETRAIN_LR_CONT, 'continuação'
    else:
        steps, lr, label = PRETRAIN_STEPS, PRETRAIN_LR, 'inicial'
    console.print(f'[bold cyan]🏋️  Treino {label}: {steps} steps  |  lr={lr}  |  params={model.n_params:,}[/bold cyan]\n')
    progresso = {'step': 0, 'loss': None, 'avg': None, 'motivo': 'limite de steps'}
    janela = deque(maxlen=LOSS_WINDOW)
    melhor = {'avg': float('inf'), 'step': 0}

    def cb(step, total, loss, loss_cot, loss_chat):
        n = step + 1
        progresso['step'] = n
        progresso['loss'] = loss
        janela.append(loss)
        avg = sum(janela) / len(janela)
        progresso['avg'] = avg
        if step % 50 == 0 or step == total - 1:
            extra = (f'  cot={loss_cot:.4f}' if loss_cot is not None else '') + \
                    (f'  chat={loss_chat:.4f}' if loss_chat is not None else '')
            console.print(f'   step {n}/{total}  loss={loss:.4f}  média({len(janela)})={avg:.4f}{extra}')
        if COT_SAMPLE_EVERY and n % COT_SAMPLE_EVERY == 0:
            if len(cot_tensor) > 0:
                _print_cot_sample(model)
            if len(chat_tensor) > 0:
                _print_chat_sample(model)
        if step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, opt, state, CHECKPOINT_LAST)
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
        pretrain(model, steps, lr, cb, opt=opt)
    except KeyboardInterrupt:
        progresso['motivo'] = 'interrupção manual / --max-minutes'
        console.print('\n[yellow]⏹️  Interrompido — salvando o que já foi treinado...[/yellow]')
    console.print('\n[yellow]💾 Salvando checkpoint final...[/yellow]')
    save_checkpoint(model, opt, state, CHECKPOINT_LAST)
    pesos = export_weights(model)
    console.print('\n[bold blue]══ RESUMO FINAL ══[/bold blue]')
    console.print(f"   Steps rodados: {progresso['step']}/{steps}")
    if progresso['loss'] is not None:
        console.print(f"   Loss final (texto): {progresso['loss']:.4f}  (média móvel: {progresso['avg']:.4f})")
    console.print(f"   Parou por: {progresso['motivo']}")
    console.print(f"   Arquitetura: {state['config']}")
    console.print(f"   Parâmetros: {state['n_params']:,}")
    console.print(f'   Checkpoint (retomar treino): [bold]{os.path.abspath(_ckpt_path(CHECKPOINT_LAST))}[/bold]  ({os.path.getsize(_ckpt_path(CHECKPOINT_LAST)) / 1e6:.0f} MB)')
    console.print(f'   Pesos fp16 (servir/distribuir): [bold]{os.path.abspath(pesos)}[/bold]  ({os.path.getsize(pesos) / 1e6:.0f} MB)\n')
    if len(cot_tensor) > 0:
        _print_cot_sample(model)
    if len(chat_tensor) > 0:
        _print_chat_sample(model)
    console.print('[bold green]✅ Treino concluído. Encerrando.[/bold green]\n')


def rodar_ask(pergunta: str, cot: bool = True, show_thoughts: bool = False, temperature: float = 0.4,
              chat: bool = False):
    """python ia_server.py --ask "What is 23 + 48?" [--show-thoughts] [--no-cot] [--chat]
    Modos: --chat = <|chat|> (conversa) | padrão = <|cot|> (raciocínio) | --no-cot = texto corrido."""
    state = novo_estado_inicial()
    model, _, modo = _carregar_ou_criar_modelo(state)
    if modo == 'novo':
        console.print('[bold yellow]⚠️  Sem checkpoint — o modelo está com pesos aleatórios. Treine primeiro.[/bold yellow]')
    model.eval()
    if chat:
        console.print(escape(generate_chat(model, pergunta, temperature=temperature)))
    elif cot:
        pensou, resp = generate_cot(model, pergunta, temperature=temperature)
        if show_thoughts:
            console.print(f'[dim]💭 {escape(pensou)}[/dim]')
        console.print(escape(resp))
    else:
        console.print(escape(generate_text(model, pergunta, temperature=temperature)))



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


# Prefetch simples de páginas da Wikipedia (só usado com --wiki / --training-local; precisa de internet).
_wiki_cache_lock = threading.Lock()
_wiki_cache: list = []       # fila de (title, texto) já buscados, prontos pra usar
_WIKI_CACHE_TARGET = 10      # quantas páginas manter prontas no buffer
_WIKI_MODE = False           # setado por run_server(wiki=True)
WIKI_CORPUS_DIR = './wiki_corpus'  # cada página vira um .txt aqui


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


def _wiki_prefetch_loop():
    console.print('[bold cyan]📖 Prefetch de conteúdo da Wikipedia iniciado (--wiki)[/bold cyan]')
    while True:
        with _wiki_cache_lock:
            precisa = len(_wiki_cache) < _WIKI_CACHE_TARGET
        if not precisa:
            time.sleep(2)
            continue
        try:
            title, texto = _wiki_random_page_text()
        except Exception as e:
            console.print(f'[bold red]⚠️  Falha ao buscar página da Wikipedia: {escape(str(e))}[/bold red]')
            time.sleep(5)
            continue
        if len(texto) <= CONTEXT_LEN + 1:
            continue
        _save_wiki_page_to_disk(title, texto)
        with _wiki_cache_lock:
            _wiki_cache.append((title, texto))
            n = len(_wiki_cache)
        console.print(f"[cyan]📖 Wiki em cache: '{escape(title)}' ({len(texto)} chars) — buffer: {n}/{_WIKI_CACHE_TARGET}[/cyan]")


# ═══════════════════════════════════════════════════════════════════════
# SERVIDOR FLASK (chat + status + download de pesos)
# ═══════════════════════════════════════════════════════════════════════
def _mensagens_para_historico(mensagens):
    """[{'role','content'}, ...] (estilo OpenAI) -> ([(user, assistant), ...], última mensagem do usuário)."""
    turnos, user_pend = [], None
    for m in mensagens:
        if not isinstance(m, dict):
            continue
        papel, txt = str(m.get('role', '')).lower(), _como_texto(m.get('content')).strip()
        if not txt:
            continue
        if papel == 'user':
            user_pend = txt if user_pend is None else user_pend + '\n' + txt
        elif papel == 'assistant' and user_pend is not None:
            turnos.append((user_pend, txt))
            user_pend = None
    return turnos, (user_pend or '')


def _build_flask_app(model, opt, state):
    app = Flask(__name__)

    @app.route('/v1/model', methods=['GET'])
    def download_model():
        # Só pesos em fp16 (~2 bytes/param) — sem estado do otimizador. Pra retomar treino use o checkpoint_last.pt do disco.
        with _server_lock:
            path = export_weights(model)
        return send_file(path, as_attachment=True, download_name=WEIGHTS_EXPORT)

    @app.route('/v1/chat', methods=['POST'])
    def chat():
        """Body JSON: prompt|message (ou messages=[{role,content},...] pra multi-turno), max_tokens (resposta),
        temperature, mode ('chat' | 'cot' | 'text'; se ausente: cot=true -> 'cot', cot=false -> 'text'),
        show_thoughts (default false), max_think_tokens."""
        data = request.get_json(force=True, silent=True) or {}
        prompt = data.get('prompt') or data.get('message') or ''
        max_new = max(1, min(int(data.get('max_tokens', 200)), CONTEXT_LEN))
        temperature = float(data.get('temperature', 0.8))
        modo = str(data.get('mode') or ('cot' if bool(data.get('cot', True)) else 'text')).lower()
        if modo not in ('chat', 'cot', 'text'):
            modo = 'cot'
        usar_cot = modo == 'cot'
        historico = []
        if isinstance(data.get('messages'), list) and data['messages']:
            historico, ultima = _mensagens_para_historico(data['messages'])
            prompt = ultima or prompt
        mostrar = bool(data.get('show_thoughts', False))
        max_think = max(1, min(int(data.get('max_think_tokens', COT_MAX_THINK_TOKENS)), CONTEXT_LEN // 2))
        with _server_lock:
            if modo == 'chat':
                pensamento, reply = '', generate_chat(model, prompt, historico=historico, max_new=max_new, temperature=temperature)
            elif usar_cot:
                pensamento, reply = generate_cot(model, prompt, max_think=max_think, max_answer=max_new, temperature=temperature)
            else:
                pensamento, reply = '', generate_text(model, prompt, max_new=max_new, temperature=temperature)
        message = {'role': 'assistant', 'content': reply}
        if usar_cot and mostrar:
            message['reasoning_content'] = pensamento
        return jsonify({'id': 'chatcmpl-local', 'object': 'chat.completion', 'model': 'tinyai-local',
                        'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}]})

    @app.route('/v1/status', methods=['GET'])
    def status():
        with _server_lock:
            return jsonify({'gen': state.get('gen', 0), 'n_params': state.get('n_params'), 'vocab_size': VOCAB,
                            'context_len': CONTEXT_LEN, 'config': model._cfg, 'cot': True,
                            'modes': ['chat', 'cot', 'text']})

    return app


# ═══════════════════════════════════════════════════════════════════════
# TREINO LOCAL POR PÁGINA (--training-local) — consome o buffer da Wikipedia
# ═══════════════════════════════════════════════════════════════════════
# Páginas recentes já treinadas (título, tensor codificado) — misturadas no batch da página
# atual pra não sobrescrever o que foi aprendido nas anteriores (catastrophic forgetting).
_replay_buffer: deque = deque(maxlen=REPLAY_BUFFER_MAX_PAGES)


def _sample_batch_misto(encoded_atual: torch.Tensor, buffer_replay: list, context_len: int, dev,
                        bs: int = REPLAY_BATCH_SIZE, fracao_replay: float = REPLAY_FRACTION):
    """Batch misturando janelas da página atual com janelas de páginas antigas do buffer
    de replay. Sempre garante pelo menos 1 exemplo da página atual."""
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
    return torch.stack(xs).long().to(dev), torch.stack(ys).long().to(dev)


def _train_step(model, opt, x, y) -> float:
    with _server_lock:
        with _amp():
            _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return loss.item()


def _local_training_direct(model: 'TinyAI', opt, state: dict,
                           loss_target: float = LOCAL_TRAIN_LOSS_TARGET,
                           max_steps_por_pagina: int = LOCAL_TRAIN_MAX_STEPS_POR_PAGINA):
    """Treino local DIRETO no modelo mestre. Entre os steps de página, com probabilidade
    COT_BATCH_FRACTION / CHAT_BATCH_FRACTION roda um step extra em dados CoT / chat (não conta pra
    meta de loss da página) — assim o modelo não desaprende a raciocinar nem a conversar."""
    console.print('[bold green]💻 Treino local direto iniciado (--training-local)[/bold green]')
    model.train()
    while True:
        with _wiki_cache_lock:
            pagina = _wiki_cache.pop(0) if _wiki_cache else None
        if pagina is None:
            time.sleep(1)
            continue
        title, texto = pagina
        encoded = _ids_to_tensor(encode(texto))
        if len(encoded) <= CONTEXT_LEN + 1:
            continue
        console.print(f"[cyan]💻 Treinando local: '{escape(title)}' ({len(texto)} chars)[/cyan]")
        step = 0
        last_loss = None
        while step < max_steps_por_pagina:
            for _tensor, _frac in ((cot_tensor, COT_BATCH_FRACTION), (chat_tensor, CHAT_BATCH_FRACTION)):
                if len(_tensor) > CONTEXT_LEN + 1 and random.random() < _frac:
                    xs, ys = _janelas(_tensor, REPLAY_BATCH_SIZE)
                    if xs:
                        _train_step(model, opt, torch.stack(xs).long().to(device), torch.stack(ys).long().to(device))
            x, y = _sample_batch_misto(encoded, list(_replay_buffer), CONTEXT_LEN, device,
                                       fracao_replay=REPLAY_FRACTION if _replay_buffer else 0.0)
            if x is None:
                break
            last_loss = _train_step(model, opt, x, y)
            step += 1
            if step == 1 or step % 10 == 0:
                console.print(f'   step {step}  loss={last_loss:.4f}')
            if last_loss <= loss_target:
                break
        with _server_lock:
            state['gen'] = state.get('gen', 0) + 1
            save_checkpoint(model, opt, state, CHECKPOINT_LAST)
        _replay_buffer.append((title, encoded))
        motivo = 'atingiu loss alvo' if last_loss is not None and last_loss <= loss_target else 'limite de steps'
        loss_txt = f'{last_loss:.4f}' if last_loss is not None else 'n/a'
        console.print(f"[bold green]✅ '{escape(title)}' concluída ({motivo}) — loss final={loss_txt}, {step} steps. "
                      f"Checkpoint salvo. (replay buffer: {len(_replay_buffer)} páginas)[/bold green]")


def run_server(wiki: bool = False, host: str = '0.0.0.0', port: int = 5000, training_local: bool = False):
    global _WIKI_MODE
    if Flask is None:
        console.print('[bold red]❌ Flask não instalado. Rode: pip install flask[/bold red]')
        return
    console.print('\n[bold blue]══ 🧬 TinyAI — Servidor Flask (denso + CoT) ══[/bold blue]')
    state = novo_estado_inicial()
    model, opt, _ = _carregar_ou_criar_modelo(state)
    if wiki or training_local:
        if training_local and not wiki:
            console.print('[bold yellow]⚠️  --training-local consome páginas do buffer da Wikipedia — ligando --wiki automaticamente.[/bold yellow]')
        _WIKI_MODE = True
        threading.Thread(target=_wiki_prefetch_loop, daemon=True).start()
    if training_local:
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        except Exception:
            pass
        threading.Thread(target=_local_training_direct, args=(model, opt, state), daemon=True).start()
    app = _build_flask_app(model, opt, state)
    extra = ' + treino local direto ligado' if training_local else ''
    console.print(f'[bold green]🚀 Servidor em http://{host}:{port}  (GET /v1/model, POST /v1/chat, GET /v1/status){extra}[/bold green]\n')
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


def _argv_val(flag, default=None):
    if flag in sys.argv:
        try:
            return sys.argv[sys.argv.index(flag) + 1]
        except IndexError:
            return default
    return default


if __name__ == '__main__':
    _setup_time_limit(sys.argv)
    if '--federated' in sys.argv:
        console.print('[bold yellow]⚠️  --federated foi removido (deprecated) — flag ignorada.[/bold yellow]')
    if '--ask' in sys.argv:
        _pergunta = _argv_val('--ask')
        if not _pergunta:
            console.print('[bold red]❌ Uso: python ia_server.py --ask "sua pergunta" [--show-thoughts] [--no-cot] [--chat][/bold red]')
        else:
            rodar_ask(_pergunta, cot='--no-cot' not in sys.argv, show_thoughts='--show-thoughts' in sys.argv,
                      chat='--chat' in sys.argv)
    elif '--serve' in sys.argv:
        try:
            _port = int(_argv_val('--port', 5000))
        except (TypeError, ValueError):
            _port = 5000
        run_server(wiki='--wiki' in sys.argv, port=_port, training_local='--training-local' in sys.argv)
    else:
        main()
