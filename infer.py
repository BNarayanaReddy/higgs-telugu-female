"""Easy inference for the fine-tuned Telugu voice — ZERO reference by default.

Text goes through the SAME frozen frontend used in training (ISO), so pass
plain Telugu / code-switch; romanization happens here.

  # local final model
  python infer.py --text "హలో, ఎలా ఉన్నారు?" --out out.wav

  # from the Hugging Face Hub
  python infer.py --repo your-username/higgs-telugu-pavani-iso \
      --text "Welcome back, ఈ రోజు కబుర్లు." --out out.wav

Optional voice-cloning still works (--ref_audio / --ref_text), but the point of
this model is that it needs no reference.
"""
import argparse
import os

import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as C
from frontend import romanize


def load(repo):
    tok = AutoTokenizer.from_pretrained(repo)
    is_lora = os.path.exists(os.path.join(repo, "adapter_config.json"))
    if is_lora:
        from peft import PeftModel, PeftConfig
        base_id = PeftConfig.from_pretrained(repo).base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(
            base_id, trust_remote_code=True, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, repo).get_base_model()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True, dtype=torch.bfloat16)
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval(), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=C.FINAL_DIR, help="local dir or HF repo id")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--scheme", default=C.ROMANIZE_SCHEME)
    ap.add_argument("--ref_audio", default=None)
    ap.add_argument("--ref_text", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    model, tok = load(args.repo)
    kw = dict(temperature=args.temperature, top_p=args.top_p)
    if args.ref_audio:
        ref, sr = torchaudio.load(args.ref_audio)
        kw.update(reference_audio=ref, reference_sample_rate=sr)
        if args.ref_text:
            kw["reference_text"] = romanize(args.ref_text, args.scheme)

    wav = model.generate_speech(romanize(args.text, args.scheme), tok, **kw)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torchaudio.save(args.out, wav.unsqueeze(0), model.config.sample_rate)
    print(f"saved {args.out}  ({wav.shape[0]/model.config.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
