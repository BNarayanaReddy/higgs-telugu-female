"""Config loader for the ISO Telugu fine-tune.

Edit `config.yaml` — this module reads it and exposes the same constants the
scripts import. Resolution order for every value:  ENV var  >  config.yaml  >  default.
So you can edit the YAML once, and still override per-run on the CLI, e.g.:
    TRAIN_MODE=lora WORK_DIR=runs/run_lora python train.py
Point at a different file with  CONFIG_YAML=/path/to/other.yaml.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load YAML if PyYAML + the file are present; otherwise fall back to env/defaults.
try:
    import yaml
    _path = os.environ.get("CONFIG_YAML", os.path.join(_HERE, "config.yaml"))
    with open(_path) as _f:
        _Y = yaml.safe_load(_f) or {}
except Exception:
    _Y = {}


def _get(key, default):
    if key in os.environ:
        return os.environ[key]                 # env value (string) wins
    if _Y.get(key) is not None:
        return _Y[key]                         # yaml value (native type)
    return default


def _s(k, d): return str(_get(k, d))
def _i(k, d): return int(_get(k, d))
def _f(k, d): return float(_get(k, d))
def _b(k, d):
    v = _get(k, d)
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
def _list_int(k):
    v = _get(k, None)
    if v in (None, "", "null", "None"):
        return None
    if isinstance(v, str):
        v = [x for x in v.replace(" ", "").split(",") if x]
    return [int(x) for x in v] or None
def _list_float(k):
    v = _get(k, None)
    if v in (None, "", "null", "None"):
        return None
    if isinstance(v, str):
        v = [x for x in v.replace(" ", "").split(",") if x]
    return [float(x) for x in v] or None
def _list_str(k, d):
    v = _get(k, d)
    return list(v) if isinstance(v, (list, tuple)) else [v]

# ─────────────────────── paths ───────────────────────
HIGGS_MODEL_DIR = _s("HIGGS_MODEL_DIR", os.path.abspath(os.path.join(_HERE, "..")))
DATASET_DIR = _s("DATASET_DIR", "/workspace/female_voice_telugu")
CSV_DIR = _s("CSV_DIR", os.path.join(DATASET_DIR, "transcripts"))
AUDIO_ROOT = _s("AUDIO_ROOT", os.path.join(DATASET_DIR, "audio"))
TRANSCRIPT_COLUMN = _s("TRANSCRIPT_COLUMN", "saaras_codemix")

WORK_DIR = _s("WORK_DIR", os.path.join(_HERE, "runs", "run1_iso"))
DATA_DIR = _s("DATA_DIR", os.path.join(WORK_DIR, "data"))     # shareable across runs
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
FINAL_DIR = os.path.join(WORK_DIR, "final_model")
SAMPLES_DIR = os.path.join(WORK_DIR, "samples")

# ─────────────────────── frontend / data ───────────────────────
ROMANIZE_SCHEME = _s("ROMANIZE_SCHEME", "iso")
SAMPLE_RATE = 24000
MAX_AUDIO_SEC = _f("MAX_AUDIO_SEC", 20.0)
MIN_AUDIO_SEC = _f("MIN_AUDIO_SEC", 0.4)
N_EVAL = _i("N_EVAL", 100)
NUM_EVAL_RENDER = _i("NUM_EVAL_RENDER", 8)

# ─────────────────────── training ───────────────────────
TRAIN_MODE = _s("TRAIN_MODE", "partial")
UNFREEZE_LAYER_INDICES = _list_int("UNFREEZE_LAYER_INDICES")
UNFREEZE_LAST_N_LAYERS = _i("UNFREEZE_LAST_N_LAYERS", 12)
TRAIN_AUDIO_EMBEDDING = _b("TRAIN_AUDIO_EMBEDDING", True)
TRAIN_FINAL_NORM = _b("TRAIN_FINAL_NORM", True)

LORA_R = _i("LORA_R", 32)
LORA_ALPHA = _i("LORA_ALPHA", 64)
LORA_DROPOUT = _f("LORA_DROPOUT", 0.05)
LORA_TARGET_MODULES = _list_str(
    "LORA_TARGET_MODULES",
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

DTYPE = _s("DTYPE", "bfloat16")
EPOCHS = _i("EPOCHS", 3)
LEARNING_RATE = _f("LEARNING_RATE", 5e-5)
WARMUP_RATIO = _f("WARMUP_RATIO", 0.03)
WEIGHT_DECAY = _f("WEIGHT_DECAY", 0.01)
GRAD_ACCUM_STEPS = _i("GRAD_ACCUM_STEPS", 16)
MAX_GRAD_NORM = _f("MAX_GRAD_NORM", 1.0)
GRADIENT_CHECKPOINTING = _b("GRADIENT_CHECKPOINTING", True)
SEED = _i("SEED", 1234)
CODEBOOK_WEIGHTS = _list_float("CODEBOOK_WEIGHTS")
RESUME = _s("RESUME", "")       # ""=fresh | "auto"=latest checkpoint | "/path/to/step_XXXXXX"

# ─────────────────────── checkpoint / listening ───────────────────────
SAVE_EVERY_STEPS = _i("SAVE_EVERY_STEPS", 500)
SAMPLE_EVERY_STEPS = _i("SAMPLE_EVERY_STEPS", 250)
KEEP_LAST_CKPTS = _i("KEEP_LAST_CKPTS", 3)
ENGLISH_CHECK_SENTENCES = _list_str("ENGLISH_CHECK_SENTENCES", [
    "The quick brown fox jumps over the lazy dog.",
    "Thanks for listening, and I will see you in the next episode.",
])

# ─────────────────────── hugging face ───────────────────────
HF_REPO_ID = _s("HF_REPO_ID", "BNarayanaReddy/higgs-telugu-female-lora-iso")
HF_PRIVATE = _b("HF_PRIVATE", True)
