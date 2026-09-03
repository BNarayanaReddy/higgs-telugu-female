"""Step 2 — fine-tune Higgs on the ISO-romanized single-speaker corpus (run #1).

Custom training loop: the model has no training-time forward, so we build the
sequence exactly like inference (model._build_prompt_ids, zero reference),
embed text with embed_tokens and audio codes with the fused audio_embedding,
run the Qwen3 backbone, apply the fused audio_head, and cross-entropy the
8 codebooks against the delay-patterned target rows (BOC ramp-in masked, EOC kept).

Full precision (bf16 native, NO quantization). Default TRAIN_MODE="partial":
unfreeze the last N layers + the fused audio embedding/head + final norm.

Single-speaker corpus => the voice is learned unconditionally; no reference and
no speaker token are needed at serve time.

  python train.py
"""
import json
import math
import os
import random
import sys

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as C


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class Utterances(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        codes = torch.load(r["codes_path"]).long()   # [T, N]
        return r["iso_text"], codes


def setup_trainable(model, mod):
    """Configure requires_grad per TRAIN_MODE. Returns (base_module, trainable_params)."""
    if C.TRAIN_MODE == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
            target_modules=C.LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        base = model.get_base_model()
        # audio embed/head are non-standard modules PEFT can't wrap — train them directly.
        if C.TRAIN_AUDIO_EMBEDDING:
            base.audio_embedding.weight.requires_grad_(True)
    else:  # partial
        base = model
        for p in model.parameters():
            p.requires_grad_(False)
        layers = model.model.layers
        if C.UNFREEZE_LAYER_INDICES:                       # data-driven (from layer_probe.py)
            idxs = sorted(i for i in C.UNFREEZE_LAYER_INDICES if 0 <= i < len(layers))
        else:                                              # analysis-aligned late block
            idxs = list(range(len(layers) - C.UNFREEZE_LAST_N_LAYERS, len(layers)))
        for i in idxs:
            for p in layers[i].parameters():
                p.requires_grad_(True)
        print(f"  unfrozen backbone layers: {idxs}", flush=True)
        if C.TRAIN_AUDIO_EMBEDDING:
            base.audio_embedding.weight.requires_grad_(True)   # audio_head is tied to this
        if C.TRAIN_FINAL_NORM:
            for p in model.model.norm.parameters():
                p.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in trainable)
    print(f"[{C.TRAIN_MODE}] trainable params: {n/1e6:.1f}M", flush=True)
    return model, base, trainable


def seq_loss(base, tok, mod, iso_text, codes_TN, device):
    """Cross-entropy over the 8 codebooks for one utterance."""
    BOC, EOC = mod.BOC_ID, mod.EOC_ID
    prompt_ids = base._build_prompt_ids(tok, iso_text, num_ref_tokens=0, reference_text=None)
    S = len(prompt_ids)

    prompt_ids_t = torch.tensor(prompt_ids, device=device)
    prompt_embeds = base.model.embed_tokens(prompt_ids_t)                 # [S, D]

    delayed = mod.apply_delay_pattern(codes_TN.to(device))               # [L, N]
    L, N = delayed.shape
    V = base.codebook_vocab_size
    audio_embeds = base.audio_embedding(delayed)                        # [L, D]

    inputs = torch.cat([prompt_embeds, audio_embeds], 0).unsqueeze(0)    # [1, S+L, D]
    hidden = base.model(inputs_embeds=inputs, use_cache=False).last_hidden_state[0]  # [S+L, D]

    # positions [S-1 .. S+L-2] predict target rows [0 .. L-1]
    pred = base.audio_head(hidden[S - 1: S + L - 1])                    # [L, N, V]
    targets = delayed.clone()
    targets[targets == BOC] = -100                                      # mask ramp-in; keep EOC

    w = C.CODEBOOK_WEIGHTS or [1.0] * N
    loss = 0.0
    for c in range(N):
        loss = loss + w[c] * F.cross_entropy(pred[:, c, :], targets[:, c], ignore_index=-100)
    return loss / sum(w)


@torch.no_grad()
def render_samples(base, tok, step, eval_texts):
    """Render held-out (unseen) Telugu lines + English forgetting-check lines."""
    base.eval()
    out = os.path.join(C.SAMPLES_DIR, f"step_{step:06d}")
    os.makedirs(out, exist_ok=True)
    # eval_texts are already ISO (from manifest_eval); English needs no romanization.
    for tag, sents in [("eval", eval_texts), ("en", C.ENGLISH_CHECK_SENTENCES)]:
        for j, s in enumerate(sents):
            try:
                wav = base.generate_speech(s, tok, temperature=0.7, top_p=0.95)
                torchaudio.save(os.path.join(out, f"{tag}_{j:02d}.wav"),
                                wav.unsqueeze(0), C.SAMPLE_RATE)
            except Exception as e:
                print(f"  sample {tag}_{j} failed: {e}", flush=True)
    base.train()
    print(f"  rendered {len(eval_texts)} held-out + {len(C.ENGLISH_CHECK_SENTENCES)} EN -> {out}", flush=True)


def save_ckpt(model, tok, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)     # partial: full model (~8GB); lora: adapter only
    tok.save_pretrained(path)
    print(f"  saved -> {path}", flush=True)
    # prune old step checkpoints to save disk (final_model is separate)
    import glob, shutil
    ckpts = sorted(glob.glob(os.path.join(C.CKPT_DIR, "step_*")))
    for old in ckpts[:-C.KEEP_LAST_CKPTS]:
        shutil.rmtree(old, ignore_errors=True)


def main():
    random.seed(C.SEED); torch.manual_seed(C.SEED)
    for d in (C.CKPT_DIR, C.SAMPLES_DIR, C.FINAL_DIR):
        os.makedirs(d, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading tokenizer + model (bf16, no quantization)...", flush=True)
    tok = AutoTokenizer.from_pretrained(C.HIGGS_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        C.HIGGS_MODEL_DIR, trust_remote_code=True, dtype=getattr(torch, C.DTYPE),
    ).to(device)
    mod = sys.modules[type(model).__module__]      # apply_delay_pattern, BOC_ID, EOC_ID
    model, base, trainable = setup_trainable(model, mod)
    if C.GRADIENT_CHECKPOINTING:
        base.model.gradient_checkpointing_enable()
    base.config.use_cache = False
    model.train()

    train_path = os.path.join(C.DATA_DIR, "manifest_train.jsonl")
    if not os.path.exists(train_path):
        train_path = os.path.join(C.DATA_DIR, "manifest.jsonl")   # fallback
    rows = load_manifest(train_path)
    ds = Utterances(rows)
    eval_path = os.path.join(C.DATA_DIR, "manifest_eval.jsonl")
    eval_texts = [r["iso_text"] for r in load_manifest(eval_path)][:C.NUM_EVAL_RENDER] \
        if os.path.exists(eval_path) else []
    print(f"corpus: {len(ds)} train utterances | {len(eval_texts)} held-out rendered each ckpt", flush=True)

    steps_per_epoch = math.ceil(len(ds) / C.GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * C.EPOCHS
    opt = torch.optim.AdamW(trainable, lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    warmup = int(total_steps * C.WARMUP_RATIO)
    def lr_at(s):
        if s < warmup:
            return s / max(1, warmup)
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    gstep = 0
    for epoch in range(C.EPOCHS):
        order = list(range(len(ds))); random.shuffle(order)
        opt.zero_grad()
        run_loss = 0.0
        for it, idx in enumerate(order):
            iso_text, codes = ds[idx]
            try:
                loss = seq_loss(base, tok, mod, iso_text, codes, device)
                (loss / C.GRAD_ACCUM_STEPS).backward()
                run_loss += loss.item()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt.zero_grad(); continue

            if (it + 1) % C.GRAD_ACCUM_STEPS == 0 or it == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(trainable, C.MAX_GRAD_NORM)
                opt.step(); sched.step(); opt.zero_grad()
                gstep += 1
                if gstep % 10 == 0:
                    avg = run_loss / (10 * C.GRAD_ACCUM_STEPS)
                    print(f"epoch {epoch} step {gstep}/{total_steps} loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
                    run_loss = 0.0
                if gstep % C.SAMPLE_EVERY_STEPS == 0:
                    render_samples(base, tok, gstep, eval_texts)
                if gstep % C.SAVE_EVERY_STEPS == 0:
                    save_ckpt(model, tok, os.path.join(C.CKPT_DIR, f"step_{gstep:06d}"))

    render_samples(base, tok, gstep, eval_texts)
    save_ckpt(model, tok, C.FINAL_DIR)
    print("TRAINING DONE. Listen to samples in:", C.SAMPLES_DIR)


if __name__ == "__main__":
    main()
