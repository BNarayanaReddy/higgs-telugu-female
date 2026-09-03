"""Download + unzip the dataset on the training server.

Default path uses the Cloudflare R2 *public* dev URL — a plain HTTPS GET, so NO
credentials are stored or needed here. Just make sure the zip's object name below
matches what you uploaded.

  # after uploading female_voice_telugu.zip to the bucket:
  DATA_DEST=/workspace python download_data.py
  export DATASET_DIR=/workspace/female_voice_telugu   # (printed at the end)
  python prepare_data.py

Private bucket instead? Do NOT hardcode keys — pass them by env and this uses the
S3 API (boto3):
  R2_ENDPOINT=https://<acct>.r2.cloudflarestorage.com R2_BUCKET=telugu-female-spotify \
  R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_ZIP_NAME=female_voice_telugu.zip \
  python download_data.py
"""
import os
import sys
import time
import zipfile

# Public R2 dev base for this bucket (a public URL, not a secret). Override via env.
R2_PUBLIC_BASE = os.environ.get(
    "R2_PUBLIC_BASE", "https://pub-fce5ae3f10a1489e961730f24b587829.r2.dev")
ZIP_NAME = os.environ.get("R2_ZIP_NAME", "female_voice_telugu.zip")
DEST = os.environ.get("DATA_DEST", "/workspace")


def _bar(done, total):
    if total <= 0:
        sys.stdout.write(f"\r  {done/1e6:.0f} MB"); sys.stdout.flush(); return
    pct = done / total
    sys.stdout.write(f"\r  [{'#'*int(40*pct):<40}] {pct*100:5.1f}%  {done/1e6:.0f}/{total/1e6:.0f} MB")
    sys.stdout.flush()


def download_public(url, out):
    import urllib.request
    print(f"GET {url}", flush=True)
    with urllib.request.urlopen(url) as r, open(out, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)); done = 0; t = time.time()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); done += len(chunk)
            if time.time() - t > 0.5:
                _bar(done, total); t = time.time()
    _bar(done, total); print()


def download_s3(out):
    import boto3
    s3 = boto3.client(
        "s3", endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    print(f"S3 GET {os.environ['R2_BUCKET']}/{ZIP_NAME}", flush=True)
    s3.download_file(os.environ["R2_BUCKET"], ZIP_NAME, out)


def main():
    os.makedirs(DEST, exist_ok=True)
    out_zip = os.path.join(DEST, ZIP_NAME)
    if not os.path.exists(out_zip):
        if os.environ.get("R2_ACCESS_KEY_ID"):
            download_s3(out_zip)
        else:
            download_public(f"{R2_PUBLIC_BASE}/{ZIP_NAME}", out_zip)
    else:
        print(f"already downloaded: {out_zip}")

    print("unzipping...", flush=True)
    with zipfile.ZipFile(out_zip) as z:
        z.extractall(DEST)

    # locate the dataset root (the dir that has transcripts/ + audio/)
    for root, dirs, _ in os.walk(DEST):
        if "transcripts" in dirs and "audio" in dirs:
            print(f"\nDATASET ready. Run:\n  export DATASET_DIR={root}")
            return
    print("WARN: could not find transcripts/+audio/ after unzip — check the zip layout.")


if __name__ == "__main__":
    main()
