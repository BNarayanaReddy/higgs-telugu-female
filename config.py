"""Loader for the single-file YAML config (`config.yaml`).

`config.yaml` is the ONE place to change anything. This module holds no editable
values — it just reads the YAML and exposes each key as a module constant,
coercing types so env-var overrides (which arrive as strings) still work.
Resolution for every value:  ENV var  >  config.yaml.

    LEARNING_RATE=1e-4 python train.py          # override one value for a run
    CONFIG_YAML=/path/to/other.yaml python ...  # use a different config file

Derived paths (CSV_DIR, AUDIO_ROOT, CKPT_DIR, FINAL_DIR, SAMPLES_DIR) are computed
from the ones above and need no YAML entry.
"""
import os

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.environ.get("CONFIG_YAML", os.path.join(_HERE, "config.yaml"))
with open(_PATH) as _f:
    _Y = yaml.safe_load(_f) or {}


class ConfigError(KeyError):
    """A required key is absent from both the environment and config.yaml."""


def _raw(key):
    """env value (string) wins over the yaml value; missing in both -> None."""
    if key in os.environ:
        return os.environ[key]
    return _Y.get(key)


def _req(key):
    v = _raw(key)
    if v is None:
        raise ConfigError(f"'{key}' missing from {_PATH} (and not set as an env var)")
    return v


def _s(k): return str(_req(k))
def _i(k): return int(_req(k))
def _f(k): return float(_req(k))
def _b(k):
    v = _req(k)
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
def _opt(k):
    v = _raw(k)
    return None if v in (None, "", "null", "None") else v
def _list_float(k):
    v = _opt(k)
    if v is None:
        return None
    if isinstance(v, str):
        v = [x for x in v.replace(" ", "").split(",") if x]
    return [float(x) for x in v] or None
def _list_str(k):
    v = _req(k)
    return list(v) if isinstance(v, (list, tuple)) else [v]

# ─────────────────────── paths ───────────────────────
HIGGS_MODEL_DIR = _s("HIGGS_MODEL_DIR")
DATASET_DIR = _s("DATASET_DIR")
TRANSCRIPT_COLUMN = _s("TRANSCRIPT_COLUMN")
WORK_DIR = _s("WORK_DIR")
DATA_DIR = _s("DATA_DIR")                                # shareable across runs

# derived (override in yaml/env only for a non-standard layout)
CSV_DIR = _opt("CSV_DIR") or os.path.join(DATASET_DIR, "transcripts")
AUDIO_ROOT = _opt("AUDIO_ROOT") or os.path.join(DATASET_DIR, "audio")
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
FINAL_DIR = os.path.join(WORK_DIR, "final_model")
SAMPLES_DIR = os.path.join(WORK_DIR, "samples")

# ─────────────────────── frontend / data ───────────────────────
ROMANIZE_SCHEME = _s("ROMANIZE_SCHEME")
SAMPLE_RATE = 24000                                      # codec rate — fixed, not a knob
MAX_AUDIO_SEC = _f("MAX_AUDIO_SEC")
MIN_AUDIO_SEC = _f("MIN_AUDIO_SEC")
N_EVAL = _i("N_EVAL")
NUM_EVAL_RENDER = _i("NUM_EVAL_RENDER")

# ─────────────────────── LoRA ───────────────────────
LORA_R = _i("LORA_R")
LORA_ALPHA = _i("LORA_ALPHA")
LORA_DROPOUT = _f("LORA_DROPOUT")
LORA_TARGET_MODULES = _list_str("LORA_TARGET_MODULES")
TRAIN_AUDIO_EMBEDDING = _b("TRAIN_AUDIO_EMBEDDING")      # the fused audio embed/head (CB0 = prosody)

# ─────────────────────── training ───────────────────────
DTYPE = _s("DTYPE")
EPOCHS = _i("EPOCHS")
LEARNING_RATE = _f("LEARNING_RATE")
WARMUP_RATIO = _f("WARMUP_RATIO")
WEIGHT_DECAY = _f("WEIGHT_DECAY")
GRAD_ACCUM_STEPS = _i("GRAD_ACCUM_STEPS")
MAX_GRAD_NORM = _f("MAX_GRAD_NORM")
GRADIENT_CHECKPOINTING = _b("GRADIENT_CHECKPOINTING")
SEED = _i("SEED")
CODEBOOK_WEIGHTS = _list_float("CODEBOOK_WEIGHTS")
RESUME = _opt("RESUME") or ""       # ""=fresh | "auto"=latest checkpoint | "/path/to/step_XXXXXX"

# ─────────────────────── checkpoint / listening ───────────────────────
SAVE_EVERY_STEPS = _i("SAVE_EVERY_STEPS")
SAMPLE_EVERY_STEPS = _i("SAMPLE_EVERY_STEPS")
KEEP_LAST_CKPTS = _i("KEEP_LAST_CKPTS")
ENGLISH_CHECK_SENTENCES = _list_str("ENGLISH_CHECK_SENTENCES")

# ─────────────────────── hugging face ───────────────────────
HF_REPO_ID = _s("HF_REPO_ID")
HF_PRIVATE = _b("HF_PRIVATE")
