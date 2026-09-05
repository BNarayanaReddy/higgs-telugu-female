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
- **Method:** **LoRA** across the backbone + the fused audio embedding/head trained
  directly, **full precision (bf16 native, NO quantization)**. The low-rank update
  regularizes, so the base's ISO-reading/prosody survives (blanket partial-FT forgot
  it). Single-speaker data ⇒ the voice is learned unconditionally, so **no reference
  and no speaker token at serve.**
- **Eval = your ears.** No automated metric harness; each checkpoint renders Telugu +
  English samples for you to listen to.

## Files

| file | what it does |
|---|---|
| `config.yaml` | all paths + hyperparameters — **the one file you edit** |
| `config.py` | loads `config.yaml` and exposes the values (no settings live here) |
| `frontend.py` | the frozen ISO romanizer (train = serve; never diverge) |
| `prepare_data.py` | romanize transcripts + encode audio → cached codes + manifest |
| `train.py` | LoRA fine-tune with the multi-codebook delay-pattern loss |
| `infer.py` | one-call zero-reference inference (local dir or HF repo) |
| `upload_hf.py` | push self-contained model + frontend + model card to HF Hub |

## Setup (server)

```bash
# 1. copy this folder AND the Higgs model repo (the parent dir with modeling_*.py,
#    config.json, tokenizer.json, model.safetensors) to the server.
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. edit config.yaml  (paths, HF repo, epochs, lr …)
nano config.yaml
```

**All settings live in `config.yaml`** — it is the single file to edit; `config.py`
just loads it. Every value can still be overridden per-run by an env var of the same
name (env > yaml), e.g. `WORK_DIR=runs/exp2 LEARNING_RATE=1e-4 python train.py`.
`HIGGS_MODEL_DIR` may be a local dir OR a HF repo id
(`multimodalart/higgs-audio-v3-tts-4b-transformers`) to auto-download the ~8 GB model.

Sanity-check the frontend before anything else: `python frontend.py`.

## Run order

```bash
python download_data.py          # pull + unzip dataset from R2 public URL → prints DATASET_DIR
export DATASET_DIR=/workspace/female_voice_telugu     # (as printed)
python prepare_data.py           # → data/manifest_{train,eval}.jsonl + codes/*.pt  (100 held out for eval)
python train.py                  # → checkpoints/, samples/step_*/  ← LISTEN to these
python upload_hf.py               # → pushes WORK_DIR/final_model to the Hub
python infer.py --repo $HF_REPO_ID --text "హలో, ఎలా ఉన్నారు?" --out out.wav
```

## What gets trained

- **LoRA:** all 36 layers, all 7 proj modules (`q/k/v/o/gate/up/down`), `r=32`. The
  low-rank update adapts the whole stack with few params, which regularizes against the
  catastrophic forgetting a blanket full-parameter fine-tune showed (US accent + flat).
- **Fused audio embedding/head:** carries CB0 = pitch/prosody (codebook-roles analysis)
  and the head is tied to it, so PEFT can't wrap it — it is unfrozen and trained directly.
- Launch a run (override any value per-run via env):
  ```bash
  WORK_DIR=runs/run_lora python train.py
  ```

## Evaluation = your ears, on held-out data

`prepare_data.py` holds out **100 utterances** (deterministic, never trained on). Each checkpoint renders `NUM_EVAL_RENDER` of them + the English forgetting-check lines into `samples/step_*/`. Because these are real held-out sentences, you can A/B the generated clip against the speaker's original recording. Judge: (1) stable, (2) authentic pronunciation, (3) prosody/pausing vs baseline, (4) English intact. Stop on the best-sounding checkpoint, not the loss.

## Hardware

- Training: a single **≥24 GB** GPU (40–80 GB comfortable). Full precision, no
  quantization. This is **not** the 6 GB laptop — that stays for listening/inference.
- Rough knobs if you OOM: keep `GRADIENT_CHECKPOINTING: true`, lower `LORA_R`, or
  lower `MAX_AUDIO_SEC`.

## Gotchas

- **Frontend parity is sacred.** Training and inference both call `romanize(text,"iso")`.
  Never romanize on one side only — it silently wrecks quality.
- **Codec decode** wants fp32 and can be memory-heavy for long clips; the modeling code
  handles fp32. On tiny GPUs decode can OOM — a non-issue on the training server.
- `audio_head.weight` is **tied** to `audio_embedding.weight`; training the embedding
  trains the head. (The load report prints it as "MISSING" — that's the tie, expected.)
- Run #1 is deliberately the smallest thing that could work. Next levers: CB0-weighted
  loss (`CODEBOOK_WEIGHTS`), a separate LR for the audio embedding, batching, longer context.
```
