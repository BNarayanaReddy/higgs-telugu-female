"""Data-driven layer selection for partial-FT (run BEFORE train.py, optional).

Answers "which layers actually matter for THIS speaker/task, not a guess." For a
sample of held-out utterances, it backprops the training loss through the whole
backbone and measures each layer's gradient magnitude per parameter (RMS) — how
strongly the objective wants to move that layer. This is the CSP-FT idea:
fine-tune the layers that carry the signal.

Output:
  WORK_DIR/layer_importance.json  {per-layer rms grad, ranked, suggested_top}
and it prints a list to paste into config.UNFREEZE_LAYER_INDICES.

  python layer_probe.py

Needs the training GPU (grads on the full backbone). Read-only wrt weights.
"""
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as C
from train import load_manifest, seq_loss

N_PROBE = 48        # held-out utterances to average importance over


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(C.HIGGS_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        C.HIGGS_MODEL_DIR, trust_remote_code=True, dtype=getattr(torch, C.DTYPE)).to(device)
    mod = sys.modules[type(model).__module__]

    for p in model.parameters():
        p.requires_grad_(False)
    for l in model.model.layers:
        for p in l.parameters():
            p.requires_grad_(True)
    model.audio_embedding.weight.requires_grad_(True)
    for p in model.model.norm.parameters():
        p.requires_grad_(True)
    model.model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    eval_path = os.path.join(C.DATA_DIR, "manifest_eval.jsonl")
    rows = load_manifest(eval_path if os.path.exists(eval_path)
                         else os.path.join(C.DATA_DIR, "manifest.jsonl"))
    random.Random(C.SEED).shuffle(rows)
    rows = rows[:N_PROBE]
    print(f"probing on {len(rows)} held-out utterances...", flush=True)

    L = len(model.model.layers)
    imp = {i: 0.0 for i in range(L)}
    audio_imp = 0.0
    used = 0
    for r in rows:
        try:
            codes = torch.load(r["codes_path"]).long()
            model.zero_grad(set_to_none=True)
            seq_loss(model, tok, mod, r["iso_text"], codes, device).backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        for i, l in enumerate(model.model.layers):
            s = n = 0
            for p in l.parameters():
                if p.grad is not None:
                    s += p.grad.float().pow(2).sum().item(); n += p.numel()
            if n:
                imp[i] += (s / n) ** 0.5
        ae = model.audio_embedding.weight.grad
        if ae is not None:
            audio_imp += (ae.float().pow(2).mean().item()) ** 0.5
        used += 1

    imp = {i: v / max(1, used) for i, v in imp.items()}
    audio_imp /= max(1, used)
    ranked = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
    topk = sorted(i for i, _ in ranked[:C.UNFREEZE_LAST_N_LAYERS])

    print("\nlayer importance (RMS grad/param), high→low:")
    for i, v in ranked:
        print(f"  L{i:2d}: {v:.3e}")
    print(f"\naudio_embedding importance: {audio_imp:.3e}")
    print(f"\n>>> paste into config.py:\nUNFREEZE_LAYER_INDICES = {topk}")

    os.makedirs(C.WORK_DIR, exist_ok=True)
    with open(os.path.join(C.WORK_DIR, "layer_importance.json"), "w") as f:
        json.dump({"per_layer_rms_grad": imp, "ranked": ranked,
                   "audio_embedding": audio_imp, "suggested_top": topk}, f, indent=2)


if __name__ == "__main__":
    main()
