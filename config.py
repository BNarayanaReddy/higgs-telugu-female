"""Central config for the ISO-romanized single-speaker Telugu fine-tune (run #1).

Edit the PATHS block for your server, then everything else follows. Keep the
romanization scheme = "iso" for run #1 (decided from the Phase-1 A/B).
"""
import os

# ─────────────────────── PATHS (edit for server) ───────────────────────
# The Higgs model repo (contains modeling_*.py, config.json, tokenizer, weights).
# Defaults to the parent of this folder (this folder lives inside the repo).
HIGGS_MODEL_DIR = os.environ.get(
    "HIGGS_MODEL_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
# Corpus: per-folder CSVs (cols: filename,start,end,old_transcript,saaras_codemix)
# and audio at  AUDIO_ROOT/<csv_stem>/<filename>.
CSV_DIR = os.environ.get("CSV_DIR", "/path/on/server/codemix_full")
AUDIO_ROOT = os.environ.get("AUDIO_ROOT", "/path/on/server/pavani_audio")
TRANSCRIPT_COLUMN = "saaras_codemix"

# Where prepared data + checkpoints + final model go.
WORK_DIR = os.environ.get("WORK_DIR", os.path.join(os.path.dirname(__file__), "runs", "run1_iso"))
DATA_DIR = os.path.join(WORK_DIR, "data")          # cached codes + manifest
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
FINAL_DIR = os.path.join(WORK_DIR, "final_model")
SAMPLES_DIR = os.path.join(WORK_DIR, "samples")    # per-checkpoint listening samples

# ─────────────────────── FRONTEND ───────────────────────
ROMANIZE_SCHEME = "iso"        # run #1 decision. ("ascii"/"native" only for ablation)

# ─────────────────────── DATA ───────────────────────
SAMPLE_RATE = 24000
MAX_AUDIO_SEC = 20.0           # skip/segment clips longer than this (memory guard)
MIN_AUDIO_SEC = 0.4

# Held-out utterances, NEVER trained on. Deterministic split (by SEED). Used both
# for the by-ear listening check and as the layer_probe.py importance set.
N_EVAL = 100
NUM_EVAL_RENDER = 8            # how many held-out lines to render each checkpoint

# ─────────────────────── TRAINING ───────────────────────
# "partial" = full-precision fine-tune of selected layers (best-fit / effectiveness
#             run). "lora" = PEFT spread thin across ALL layers (efficiency run:
#             "how cheaply can we adapt the base for one speaker"). Both train the
#             fused audio embedding/head directly (non-standard module PEFT can't wrap).
TRAIN_MODE = os.environ.get("TRAIN_MODE", "partial")

# WHICH layers to unfreeze in partial mode — analysis-aligned, not arbitrary:
#   * audio embedding/head: carries CB0 = pitch/prosody (codebook-roles analysis).
#   * late block: the reference/expression effect concentrates in late layers
#     (ref_influence_decay: with-ref vs no-ref L2 explodes ~L29-35).
# For a DATA-DRIVEN list, run layer_probe.py and paste its ranked indices into
# UNFREEZE_LAYER_INDICES (overrides UNFREEZE_LAST_N_LAYERS). None => use last-N.
UNFREEZE_LAYER_INDICES = None      # e.g. [35,34,33,32,31,30,29,28,12,11] from the probe
UNFREEZE_LAST_N_LAYERS = 12        # fallback when UNFREEZE_LAYER_INDICES is None
TRAIN_AUDIO_EMBEDDING = True        # the [8208,2560] fused embed/head — where expression lives
TRAIN_FINAL_NORM = True

# lora mode (only if TRAIN_MODE="lora"): backbone target modules.
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DTYPE = "bfloat16"             # model's native precision; NO quantization (per decision)
EPOCHS = 3
LEARNING_RATE = 1e-4          # partial-FT; use ~2e-4 for lora
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
GRAD_ACCUM_STEPS = 16          # effective batch (per-example micro-batch = 1)
MAX_GRAD_NORM = 1.0
GRADIENT_CHECKPOINTING = True
SEED = 1234

# Loss over the 8 codebooks. Equal weight for run #1; CB0 upweighting is a run #2
# knob (CB0 carries pitch/prosody — see analysis). BOC ramp-in is masked; EOC kept.
CODEBOOK_WEIGHTS = None        # None = equal. e.g. [1.5,1,1,1,1,1,1,1] to favor CB0.

# ─────────────────────── CHECKPOINT / LISTENING ───────────────────────
SAVE_EVERY_STEPS = 500
SAMPLE_EVERY_STEPS = 500       # render zero-reference samples for listening
# Sentences rendered each checkpoint (put your held-out, UNSEEN lines here:
# questions, exclamations, long-form, heavy code-switch). Native script OK —
# they are romanized through the same frontend at generation time.
LISTEN_SENTENCES = [
    "హలో, మీరు ఎలా ఉన్నారు? ఈ రోజు చాలా బాగుంది కదా!",
    "నేను చెప్పేది కొంచెం జాగ్రత్తగా వినండి... ఇది చాలా ముఖ్యమైన విషయం.",
    "Welcome back to the show, ఈ రోజు మనం ఒక interesting topic గురించి మాట్లాడుకుందాం.",
]
# A few English lines to catch base forgetting (listen by ear each checkpoint).
ENGLISH_CHECK_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Thanks for listening, and I will see you in the next episode.",
]

# ─────────────────────── HF HUB ───────────────────────
HF_REPO_ID = os.environ.get("HF_REPO_ID", "BNarayaanReddy/higgs-telugu-female-iso")
HF_PRIVATE = True
