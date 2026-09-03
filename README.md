# Telugu ISO Fine-tune — Higgs Audio v3 (run #1)

Turn Higgs Audio v3 (4B) into an **expressive single-speaker Telugu voice avatar**
that generates from text **with no reference clip**. This folder is self-contained
and made to be copied to a training server and run there.

## Why this shape (decisions already made)

- **Root cause:** native Telugu tokenizes into UTF-8 byte-fragments (~0.7 chars/token
  vs 4.6 for English), so the model can't plan prosody → flat, dictation-like output.
  The codec preserves expression fully; the gap is in the LM's code prediction.
- **Fix = frontend + adaptation.** We romanize Telugu to **ISO-15919** (`frontend.py`)
  so it rides English's token paths, then fine-tune on the corpus. Phase-1 A/B chose
  **ISO** (keeps every phoneme, least accent; ASCII bakes in an English accent, native
  stays byte-soup). Zero-training romanization is unstable — training is what fixes it.
- **Method:** targeted **partial fine-tuning** (last N layers + the fused audio
  embedding/head + final norm), **full precision (bf16 native, NO quantization)**.
  Single-speaker data ⇒ the voice is learned unconditionally, so **no reference and no
  speaker token at serve.**
- **Eval = your ears.** No automated metric harness; each checkpoint renders Telugu +
  English samples for you to listen to.

## Files

| file | what it does |
|---|---|
| `config.py` | all paths + hyperparameters — **edit this first** |
| `frontend.py` | the frozen ISO romanizer (train = serve; never diverge) |
| `prepare_data.py` | romanize transcripts + encode audio → cached codes + manifest |
| `train.py` | partial-FT (or LoRA) with the multi-codebook delay-pattern loss |
| `infer.py` | one-call zero-reference inference (local dir or HF repo) |
| `upload_hf.py` | push self-contained model + frontend + model card to HF Hub |

## Setup (server)

```bash
# 1. copy this folder AND the Higgs model repo (the parent dir with modeling_*.py,
#    config.json, tokenizer.json, model.safetensors) to the server.
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. point config.py at your data (or export env vars):
# HIGGS_MODEL_DIR can be a local repo dir OR a HF repo id to auto-download the ~8GB model:
export HIGGS_MODEL_DIR=multimodalart/higgs-audio-v3-tts-4b-transformers
export DATASET_DIR=/workspace/female_voice_telugu     # set by download_data.py (unzip target)
export WORK_DIR=/abs/path/to/runs/run1_iso
export HF_REPO_ID=BNarayanaReddy/higgs-telugu-pavani-iso
```

Sanity-check the frontend before anything else: `python frontend.py`.

## Run order

```bash
python download_data.py          # pull + unzip dataset from R2 public URL → prints DATASET_DIR
export DATASET_DIR=/workspace/female_voice_telugu     # (as printed)
python prepare_data.py           # → data/manifest_{train,eval}.jsonl + codes/*.pt  (100 held out for eval)
python layer_probe.py            # (optional, partial-FT) rank layers by importance → paste UNFREEZE_LAYER_INDICES
python train.py                  # → checkpoints/, samples/step_*/  ← LISTEN to these
python upload_hf.py               # → pushes WORK_DIR/final_model to the Hub
python infer.py --repo $HF_REPO_ID --text "హలో, ఎలా ఉన్నారు?" --out out.wav
```

## The two runs (both planned)

| run | `TRAIN_MODE` | question it answers | design |
|---|---|---|---|
| **efficiency** | `lora` | how *cheaply* can the base adapt to one speaker? | LoRA **spread thin across all 36 layers** (r=32) + the audio embedding — broad & light **by intent** |
| **effectiveness** | `partial` | best quality we can reach | full-precision FT of the **most important layers** (from `layer_probe.py`) + audio embedding |

```bash
TRAIN_MODE=lora    WORK_DIR=runs/run_lora    python train.py
TRAIN_MODE=partial WORK_DIR=runs/run_partial python train.py
```

## Layer selection — analysis-driven, not arbitrary

- **Audio embedding/head (both runs):** carries CB0 = pitch/prosody (codebook-roles analysis); the head is tied to it. Always trained.
- **Partial-FT layers:** the reference/expression effect concentrates in **late layers** (`ref_influence_decay`: with-ref vs no-ref L2 explodes ~L29–35), so the default is a late block. **To make it fully data-driven, run `layer_probe.py`** — it backprops the loss on the 100 held-out utterances and ranks layers by gradient magnitude, then prints an `UNFREEZE_LAYER_INDICES` list to paste into `config.py`. That converts "last 12" from a heuristic into a measurement.
- **LoRA layers:** all layers, all 7 proj modules — the right design for the *efficiency* question (adjust the whole stack with minimal params).

## Evaluation = your ears, on held-out data

`prepare_data.py` holds out **100 utterances** (deterministic, never trained on). Each checkpoint renders `NUM_EVAL_RENDER` of them + the English forgetting-check lines into `samples/step_*/`. Because these are real held-out sentences, you can A/B the generated clip against the speaker's original recording. Judge: (1) stable, (2) authentic pronunciation, (3) prosody/pausing vs baseline, (4) English intact. Stop on the best-sounding checkpoint, not the loss.

## Hardware

- Training: a single **≥24 GB** GPU (40–80 GB comfortable). Full precision, no
  quantization. This is **not** the 6 GB laptop — that stays for listening/inference.
- Rough knobs if you OOM: lower `UNFREEZE_LAST_N_LAYERS`, keep
  `GRADIENT_CHECKPOINTING=True`, or switch `TRAIN_MODE="lora"`.

## Gotchas

- **Frontend parity is sacred.** Training and inference both call `romanize(text,"iso")`.
  Never romanize on one side only — it silently wrecks quality.
- **Codec decode** wants fp32 and can be memory-heavy for long clips; the modeling code
  handles fp32. On tiny GPUs decode can OOM — a non-issue on the training server.
- `audio_head.weight` is **tied** to `audio_embedding.weight`; training the embedding
  trains the head. (The load report prints it as "MISSING" — that's the tie, expected.)
- Run #1 is deliberately the smallest thing that could work. Next runs: CB0-weighted
  loss, partial-FT vs LoRA A/B, an optional `native` control, longer context.
```
