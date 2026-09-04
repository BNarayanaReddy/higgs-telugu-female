"""Remove corrupt / ASR-failure rows from an EXISTING manifest — no full re-prep.

Applies the same precise valid_transcript() as prepare_data.py to each row's
iso_text, backs up the original to <manifest>.bak (once), and rewrites the
manifest with only the good rows. Cached .pt code files for dropped rows are
left on disk (unused, harmless).

  python clean_manifest.py                              # cleans DATA_DIR/manifest_{train,eval}.jsonl
  DATA_DIR=/dev/shm/data python clean_manifest.py
  python clean_manifest.py /path/a.jsonl /path/b.jsonl # explicit files
"""
import json
import os
import shutil
import sys

import config as C
from prepare_data import valid_transcript


def clean(path):
    if not os.path.exists(path):
        print(f"skip (not found): {path}")
        return
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    good, removed = [], []
    for r in rows:
        (good if valid_transcript(r.get("iso_text", "")) else removed).append(r)

    if not removed:
        print(f"{os.path.basename(path)}: clean already ({len(rows)} rows)")
        return

    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)          # keep the original once
    with open(path, "w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{os.path.basename(path)}: {len(rows)} -> {len(good)}  (removed {len(removed)}); backup: {bak}")
    for r in removed[:10]:
        print(f"    - {r.get('id')}: {r.get('iso_text','')[:60]!r}")


def main():
    targets = sys.argv[1:] or [
        os.path.join(C.DATA_DIR, "manifest_train.jsonl"),
        os.path.join(C.DATA_DIR, "manifest_eval.jsonl"),
    ]
    for p in targets:
        clean(p)


if __name__ == "__main__":
    main()
