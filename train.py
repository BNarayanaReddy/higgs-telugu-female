"""Step 2 — fine-tune Higgs on the ISO-romanized single-speaker corpus.

Custom training loop: the model has no training-time forward, so we build the
sequence exactly like inference (base._build_prompt_ids, zero reference), embed
text with embed_tokens and audio codes with the fused audio_embedding, run the
Qwen3 backbone, apply the fused audio_head, and cross-entropy the 8 codebooks
against the delay-patterned target rows (BOC ramp-in masked, EOC kept).

Full precision (bf16 native, NO quantization). TRAIN_MODE="partial" unfreezes
selected layers + the fused audio embedding/head + final norm; "lora" adds LoRA
across the backbone and trains the audio embedding directly.

Features: held-out eval loss + listening samples every SAMPLE_EVERY_STEPS;
checkpoints (with optimizer/scheduler/step/RNG) every SAVE_EVERY_STEPS, prunable;
resume with RESUME=auto (latest) or RESUME=/path/to/checkpoint.

  python train.py
"""
import glob
import json
import math
import os
import random
import shutil
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
        return r["iso_text"], torch.load(r["codes_path"]).long()   # iso_text, [T,N]


def seq_loss(base, tok, mod, iso_text, codes_TN, device):
    """Cross-entropy over the 8 codebooks for one utterance."""
    BOC = mod.BOC_ID
    prompt_ids = base._build_prompt_ids(tok, iso_text, num_ref_tokens=0, reference_text=None)
    S = len(prompt_ids)
    prompt_embeds = base.model.embed_tokens(torch.tensor(prompt_ids, device=device))  # [S,D]

    delayed = mod.apply_delay_pattern(codes_TN.to(device))        # [L,N]
    L, N = delayed.shape
    V = base.codebook_vocab_size
    audio_embeds = base.audio_embedding(delayed)                 # [L,D]

    inputs = torch.cat([prompt_embeds, audio_embeds], 0).unsqueeze(0)          # [1,S+L,D]
    hidden = base.model(inputs_embeds=inputs, use_cache=False).last_hidden_state[0]
    pred = base.audio_head(hidden[S - 1: S + L - 1])             # [L,N,V] predicts rows 0..L-1
    targets = delayed.clone()
    targets[targets == BOC] = -100                              # mask ramp-in; keep EOC

    w = C.CODEBOOK_WEIGHTS or [1.0] * N
    loss = 0.0
    for c in range(N):
        loss = loss + w[c] * F.cross_entropy(pred[:, c, :], targets[:, c], ignore_index=-100)
    return loss / sum(w)


@torch.no_grad()
def eval_loss(base, tok, mod, eval_rows, device):
    base.eval()
    tot = n = 0
    for r in eval_rows:
        try:
            codes = torch.load(r["codes_path"]).long()
            tot += seq_loss(base, tok, mod, r["iso_text"], codes, device).item(); n += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
    base.train()
    return tot / max(1, n)


@torch.no_grad()
def render_samples(base, tok, step, eval_texts):
    base.eval()
    out = os.path.join(C.SAMPLES_DIR, f"step_{step:06d}")
    os.makedirs(out, exist_ok=True)
    for tag, sents in [("eval", eval_texts), ("en", C.ENGLISH_CHECK_SENTENCES)]:
        for j, s in enumerate(sents):
            try:
                wav = base.generate_speech(s, tok, temperature=0.7, top_p=0.95)
                torchaudio.save(os.path.join(out, f"{tag}_{j:02d}.wav"), wav.unsqueeze(0), C.SAMPLE_RATE)
            except Exception as e:
                print(f"  sample {tag}_{j} failed: {e}", flush=True)
    base.train()
    print(f"  rendered {len(eval_texts)} held-out + {len(C.ENGLISH_CHECK_SENTENCES)} EN -> {out}", flush=True)


def resolve_resume():
    r = (C.RESUME or "").strip()
    if not r or r.lower() in ("none", "false"):
        return None
    if r == "auto":
        cks = [c for c in sorted(glob.glob(os.path.join(C.CKPT_DIR, "step_*")))
               if os.path.exists(os.path.join(c, "training_state.pt"))]
        return cks[-1] if cks else None
    return r if os.path.exists(r) else None


def build_model(device):
    """Load model (mode + resume aware), set requires_grad. Returns model, base, resume_dir."""
    dtype = getattr(torch, C.DTYPE)
    resume_dir = resolve_resume()

    if C.TRAIN_MODE == "lora":
        base_model = AutoModelForCausalLM.from_pretrained(
            C.HIGGS_MODEL_DIR, trust_remote_code=True, dtype=dtype).to(device)
        if resume_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
        else:
            from peft import LoraConfig, get_peft_model
            # No task_type: the Higgs model has no HF generate interface
            # (prepare_inputs_for_generation); a generic PeftModel just injects LoRA,
            # and we run our own forward + generate_speech.
            model = get_peft_model(base_model, LoraConfig(
                r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
                target_modules=C.LORA_TARGET_MODULES, bias="none"))
        base = model.get_base_model()
        if C.TRAIN_AUDIO_EMBEDDING:
            base.audio_embedding.weight.requires_grad_(True)     # PEFT can't wrap it → train directly
        if resume_dir:
            ae = os.path.join(resume_dir, "audio_embedding.pt")
            if os.path.exists(ae):
                base.audio_embedding.weight.data.copy_(torch.load(ae, map_location=device))
    else:  # partial
        model = AutoModelForCausalLM.from_pretrained(
            resume_dir or C.HIGGS_MODEL_DIR, trust_remote_code=True, dtype=dtype).to(device)
        base = model
        for p in model.parameters():
            p.requires_grad_(False)
        layers = model.model.layers
        if C.UNFREEZE_LAYER_INDICES:
            idxs = sorted(i for i in C.UNFREEZE_LAYER_INDICES if 0 <= i < len(layers))
        else:
            idxs = list(range(len(layers) - C.UNFREEZE_LAST_N_LAYERS, len(layers)))
        for i in idxs:
            for p in layers[i].parameters():
                p.requires_grad_(True)
        print(f"  unfrozen backbone layers: {idxs}", flush=True)
        if C.TRAIN_AUDIO_EMBEDDING:
            base.audio_embedding.weight.requires_grad_(True)     # audio_head is tied to this
        if C.TRAIN_FINAL_NORM:
            for p in model.model.norm.parameters():
                p.requires_grad_(True)

    # transformers >=5.5 expects _tied_weights_keys as a dict {tied: source}; the ported
    # model declares a list, which crashes save_pretrained at tied.keys(). Normalize it.
    if isinstance(getattr(type(base), "_tied_weights_keys", None), list):
        type(base)._tied_weights_keys = {"audio_head.weight": "audio_embedding.weight"}

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{C.TRAIN_MODE}] trainable params: {n/1e6:.1f}M"
          + (f"  (RESUMED from {resume_dir})" if resume_dir else ""), flush=True)
    return model, base, resume_dir


def save_ckpt(model, base, tok, path, state=None):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)                               # partial: full model; lora: adapter
    tok.save_pretrained(path)
    torch.save(base.audio_embedding.weight.data.detach().cpu(),
               os.path.join(path, "audio_embedding.pt"))       # trained separately in lora mode
    if state is not None:
        torch.save(state, os.path.join(path, "training_state.pt"))
    print(f"  saved -> {path}", flush=True)
    for old in sorted(glob.glob(os.path.join(C.CKPT_DIR, "step_*")))[:-C.KEEP_LAST_CKPTS]:
        shutil.rmtree(old, ignore_errors=True)


def save_final(model, base, tok):
    os.makedirs(C.FINAL_DIR, exist_ok=True)
    if C.TRAIN_MODE == "lora":
        merged = model.merge_and_unload()                     # bake LoRA into a self-contained model
        merged.save_pretrained(C.FINAL_DIR)                    # includes the trained audio_embedding
    else:
        model.save_pretrained(C.FINAL_DIR)
    tok.save_pretrained(C.FINAL_DIR)
    print(f"  final model -> {C.FINAL_DIR}", flush=True)


def main():
    random.seed(C.SEED); torch.manual_seed(C.SEED)
    for d in (C.CKPT_DIR, C.SAMPLES_DIR, C.FINAL_DIR):
        os.makedirs(d, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading tokenizer + model (bf16, no quantization)...", flush=True)
    tok = AutoTokenizer.from_pretrained(C.HIGGS_MODEL_DIR)
    model, base, resume_dir = build_model(device)
    mod = sys.modules[type(base).__module__]                  # apply_delay_pattern, BOC_ID, EOC_ID
    if C.GRADIENT_CHECKPOINTING:
        base.model.gradient_checkpointing_enable()
    base.config.use_cache = False
    model.train()

    train_path = os.path.join(C.DATA_DIR, "manifest_train.jsonl")
    if not os.path.exists(train_path):
        train_path = os.path.join(C.DATA_DIR, "manifest.jsonl")
    ds = Utterances(load_manifest(train_path))
    eval_path = os.path.join(C.DATA_DIR, "manifest_eval.jsonl")
    eval_rows = load_manifest(eval_path) if os.path.exists(eval_path) else []
    eval_texts = [r["iso_text"] for r in eval_rows][:C.NUM_EVAL_RENDER]
    print(f"corpus: {len(ds)} train | {len(eval_rows)} held-out (eval loss) | "
          f"{len(eval_texts)} rendered/ckpt", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    steps_per_epoch = math.ceil(len(ds) / C.GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * C.EPOCHS
    opt = torch.optim.AdamW(trainable, lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    warmup = int(total_steps * C.WARMUP_RATIO)
    def lr_at(s):
        if s < warmup:
            return s / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    # ---- resume optimizer/scheduler/step/RNG ----
    start_epoch = start_it = gstep = 0
    resume_order = None
    if resume_dir and os.path.exists(os.path.join(resume_dir, "training_state.pt")):
        st = torch.load(os.path.join(resume_dir, "training_state.pt"), map_location="cpu")
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        gstep = st["gstep"]; start_epoch = st["epoch"]; start_it = st["it"] + 1
        resume_order = st["order"]
        random.setstate(st["py_rng"]); torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        print(f"resumed at epoch {start_epoch}, it {start_it}, gstep {gstep}", flush=True)

    for epoch in range(start_epoch, C.EPOCHS):
        if epoch == start_epoch and resume_order is not None:
            order, begin = resume_order, start_it
        else:
            order = list(range(len(ds))); random.shuffle(order); begin = 0
        opt.zero_grad(); run_loss = 0.0
        for it in range(begin, len(order)):
            iso_text, codes = ds[order[it]]
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
                    print(f"epoch {epoch} step {gstep}/{total_steps} "
                          f"loss {run_loss/(10*C.GRAD_ACCUM_STEPS):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
                    run_loss = 0.0
                if gstep % C.SAMPLE_EVERY_STEPS == 0:
                    if eval_rows:
                        print(f"  [eval] held-out loss {eval_loss(base, tok, mod, eval_rows, device):.4f}", flush=True)
                    render_samples(base, tok, gstep, eval_texts)
                if gstep % C.SAVE_EVERY_STEPS == 0:
                    state = dict(opt=opt.state_dict(), sched=sched.state_dict(), gstep=gstep,
                                 epoch=epoch, it=it, order=order,
                                 py_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                                 cuda_rng=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
                    save_ckpt(model, base, tok, os.path.join(C.CKPT_DIR, f"step_{gstep:06d}"), state)

    render_samples(base, tok, gstep, eval_texts)
    save_final(model, base, tok)
    print("TRAINING DONE. Listen to samples in:", C.SAMPLES_DIR)


if __name__ == "__main__":
    main()
