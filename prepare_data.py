"""Step 1 — prepare the corpus for training.

For every row across all CSVs in CSV_DIR:
  * romanize the transcript with the frozen frontend (ISO),
  * load + resample the audio, skip too-short/too-long clips,
  * encode to 8-codebook codes with the Higgs v2 codec,
  * cache codes to disk and append to a manifest.

Output:
  DATA_DIR/manifest.jsonl   one line per utterance: {id, iso_text, codes_path, dur}
  DATA_DIR/codes/<id>.pt    int16 tensor [T, 8]

Run once on the server (needs internet the first time to fetch the codec):
  python prepare_data.py
"""
import csv
import glob
import json
import os
import random

import torch
import torchaudio
from transformers import AutoModel

import config as C
from frontend import romanize


def load_codec(device):
    codec = AutoModel.from_pretrained(
        "bosonai/higgs-audio-v2-tokenizer", trust_remote_code=True, dtype=torch.float32
    ).to(device).eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    return codec


@torch.no_grad()
def encode(codec, wav, sr, device):
    if sr != C.SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, C.SAMPLE_RATE)
    if wav.ndim == 2:
        wav = wav.mean(0, keepdim=True)          # mono
    wav = wav.unsqueeze(0).to(device, torch.float32)  # [1,1,L]
    codes = codec.encode(wav).audio_codes         # [1, N, T]
    return codes.squeeze(0).transpose(0, 1).contiguous().cpu()  # [T, N]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(C.DATA_DIR, "codes"), exist_ok=True)
    codec = load_codec(device)

    entries = []          # collect small manifest dicts; codes stream to disk
    n_ok = n_skip = 0

    for csv_path in sorted(glob.glob(os.path.join(C.CSV_DIR, "*.csv"))):
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fn = row["filename"]
                text = (row.get(C.TRANSCRIPT_COLUMN) or "").strip()
                audio_path = os.path.join(C.AUDIO_ROOT, stem, fn)
                if not text or not os.path.exists(audio_path):
                    n_skip += 1
                    continue
                try:
                    wav, sr = torchaudio.load(audio_path)
                except Exception:
                    n_skip += 1
                    continue
                dur = wav.shape[-1] / sr
                if dur < C.MIN_AUDIO_SEC or dur > C.MAX_AUDIO_SEC:
                    n_skip += 1
                    continue

                uid = f"{stem}__{os.path.splitext(fn)[0]}"
                codes = encode(codec, wav, sr, device).to(torch.int16)
                codes_path = os.path.join(C.DATA_DIR, "codes", f"{uid}.pt")
                torch.save(codes, codes_path)

                entries.append({
                    "id": uid,
                    "iso_text": romanize(text, C.ROMANIZE_SCHEME),
                    "codes_path": codes_path,
                    "dur": round(dur, 2),
                })
                n_ok += 1
                if n_ok % 200 == 0:
                    print(f"  prepared {n_ok} (skipped {n_skip})", flush=True)

    # deterministic train / eval split — eval is held out from training entirely
    random.Random(C.SEED).shuffle(entries)
    n_eval = min(C.N_EVAL, max(0, len(entries) // 10))   # never take >10% as eval
    eval_rows, train_rows = entries[:n_eval], entries[n_eval:]

    def dump(name, rows):
        with open(os.path.join(C.DATA_DIR, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("manifest.jsonl", entries)          # all, for reference
    dump("manifest_train.jsonl", train_rows)
    dump("manifest_eval.jsonl", eval_rows)
    print(f"DONE. {n_ok} prepared ({n_skip} skipped) -> "
          f"{len(train_rows)} train / {len(eval_rows)} eval (held out).")
    print(f"Manifests in {C.DATA_DIR}")


if __name__ == "__main__":
    main()
