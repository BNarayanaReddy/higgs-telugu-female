"""Step 3 — push the fine-tuned model to the Hugging Face Hub.

Makes the model self-contained and easy to infer from anywhere: copies the
custom modeling code + the frozen frontend into the repo and writes a model card
with a runnable snippet.

  huggingface-cli login          # once
  python upload_hf.py            # uses HF_REPO_ID from config.yaml (or env)

Uploads WORK_DIR/final_model — the merged, self-contained model (LoRA already
baked in via merge_and_unload), so it loads anywhere with no adapter needed.
"""
import os
import shutil

from huggingface_hub import HfApi, upload_folder

import config as C

MODEL_CARD = """---
license: other
language: [te, en]
pipeline_tag: text-to-speech
tags: [text-to-speech, telugu, higgs-audio, voice-avatar, code-switch]
base_model: bosonai/higgs-audio-v3-tts-4b
---

# Higgs Audio v3 — Telugu single-speaker voice (ISO-romanized fine-tune)

Expressive single-speaker **Telugu** (code-switch) TTS, fine-tuned from Higgs
Audio v3 (4B). Generates this speaker's voice **from text with no reference clip**.

**Frontend matters:** Telugu text must be ISO-15919 romanized with the *same*
`frontend.py` used in training (Telugu tokenizes to byte-fragments otherwise).
`generate_speech` handles the rest.

```python
import torch, torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
from frontend import romanize          # shipped in this repo

repo = "{repo}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True,
                                             dtype=torch.bfloat16).to("cuda").eval()

text = "హలో, ఈ రోజు ఎలా ఉన్నారు?"
wav = model.generate_speech(romanize(text, "iso"), tok, temperature=0.7, top_p=0.95)
torchaudio.save("out.wav", wav.unsqueeze(0), model.config.sample_rate)
```

Run #1 of an ISO-romanized single-speaker adaptation. Research use.
"""


def ensure_custom_code(dst):
    """Copy the trust_remote_code files + the frontend into the upload dir."""
    for name in ("modeling_higgs_multimodal_qwen3.py",
                 "configuration_higgs_multimodal_qwen3.py",
                 "chat_template.jinja"):
        src = os.path.join(C.HIGGS_MODEL_DIR, name)
        if os.path.exists(src) and not os.path.exists(os.path.join(dst, name)):
            shutil.copy2(src, dst)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "frontend.py"),
                 os.path.join(dst, "frontend.py"))


def main():
    src = C.FINAL_DIR
    assert os.path.isdir(src), f"no final model at {src} — train first"
    ensure_custom_code(src)
    with open(os.path.join(src, "README.md"), "w", encoding="utf-8") as f:
        f.write(MODEL_CARD.format(repo=C.HF_REPO_ID))

    HfApi().create_repo(C.HF_REPO_ID, private=C.HF_PRIVATE, exist_ok=True)
    upload_folder(repo_id=C.HF_REPO_ID, folder_path=src,
                  commit_message="Telugu ISO single-speaker fine-tune (run LoRA)")
    print(f"pushed -> https://huggingface.co/{C.HF_REPO_ID}")


if __name__ == "__main__":
    main()
