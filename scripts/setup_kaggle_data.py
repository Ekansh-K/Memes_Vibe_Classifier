#!/usr/bin/env python3
"""Download & extract MMHS dataset from Kaggle into ./dataset, then verify.

Dataset: https://www.kaggle.com/datasets/ekanshkhullar/updated-hate-speech-dataset

Auth (any of):
  - KAGGLE_API_TOKEN env  (new KGAT_… tokens)  + optional KAGGLE_USERNAME
  - ~/.kaggle/access_token
  - KAGGLE_USERNAME + KAGGLE_KEY  /  ~/.kaggle/kaggle.json (legacy)

Usage:
  source scripts/remote_secrets.env
  bash scripts/setup_kaggle_auth.sh
  python scripts/setup_kaggle_data.py
  python scripts/setup_kaggle_data.py --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "ekanshkhullar/updated-hate-speech-dataset"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [kaggle-data] {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [kaggle-data] ✓ {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [kaggle-data] ✗ {msg}", flush=True)


def _load_secrets_file() -> None:
    secrets = PROJECT_ROOT / "scripts" / "remote_secrets.env"
    if not secrets.exists():
        return
    log(f"Loading {secrets}")
    for line in secrets.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


def _install_token_files() -> None:
    """Write ~/.kaggle/access_token from env if present."""
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        return
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_path = kaggle_dir / "access_token"
    token_path.write_text(token, encoding="utf-8")
    token_path.chmod(0o600)
    log(f"Wrote {token_path} (token prefix {token[:8]}…)")


def _have_kaggle_creds() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if (Path.home() / ".kaggle" / "access_token").exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def _run(cmd: list[str]) -> None:
    log("Running: " + " ".join(cmd))
    env = os.environ.copy()
    # Ensure token visible to kaggle CLI
    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.exists() and "KAGGLE_API_TOKEN" not in env:
        env["KAGGLE_API_TOKEN"] = token_file.read_text(encoding="utf-8").strip()
    subprocess.check_call(cmd, env=env)


def main() -> int:
    _load_secrets_file()
    _install_token_files()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--slug",
        default=os.environ.get("KAGGLE_DATASET_SLUG", DEFAULT_SLUG),
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("MMHS_DATA_DIR", str(PROJECT_ROOT / "dataset"))),
        help="Final dataset directory",
    )
    ap.add_argument("--force", action="store_true", help="Re-download even if present")
    ap.add_argument("--skip-verify", action="store_true", help="Skip verify_dataset.py")
    args = ap.parse_args()

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "MMHS150K_GT.json"

    log(f"Dataset slug : {args.slug}")
    log(f"Dest         : {dest}")
    log(f"User         : {os.environ.get('KAGGLE_USERNAME', '(not set)')}")
    log(f"Token set    : {'yes' if os.environ.get('KAGGLE_API_TOKEN') or (Path.home()/'.kaggle'/'access_token').exists() else 'no'}")

    if marker.exists() and not args.force:
        ok(f"Dataset already present: {marker}")
        if not args.skip_verify:
            return _verify(dest)
        return 0

    if not _have_kaggle_creds():
        err("Kaggle credentials not found.")
        err("  source scripts/remote_secrets.env")
        err("  bash scripts/setup_kaggle_auth.sh")
        return 1

    tmp = PROJECT_ROOT / ".kaggle_download"
    if tmp.exists():
        log(f"Cleaning old temp dir {tmp}")
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    log("Starting Kaggle download (this can take a long time — images are large)…")
    t0 = time.time()
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.slug,
        "-p",
        str(tmp),
        "--unzip",
    ]
    try:
        _run(cmd)
    except FileNotFoundError:
        log("python -m kaggle failed; trying `kaggle` binary…")
        _run(["kaggle", "datasets", "download", "-d", args.slug, "-p", str(tmp), "--unzip"])
    except subprocess.CalledProcessError as e:
        err(f"Kaggle download failed (exit {e.returncode})")
        err("Open the dataset page in a browser and Accept terms once:")
        err(f"  https://www.kaggle.com/datasets/{args.slug}")
        err("Also: pip install -U kaggle")
        return 1

    log(f"Download+unzip finished in {(time.time()-t0)/60:.1f} min")

    gt_candidates = list(tmp.rglob("MMHS150K_GT.json"))
    if not gt_candidates:
        err("MMHS150K_GT.json not found in download. Listing files:")
        for p in sorted(tmp.rglob("*"))[:80]:
            if p.is_file():
                log(f"  {p.relative_to(tmp)}")
        return 1

    src_root = gt_candidates[0].parent
    ok(f"Found data root: {src_root}")

    log("Moving files into dest…")
    for item in src_root.iterdir():
        target = dest / item.name
        log(f"  → {item.name}")
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))

    # Move any leftover top-level useful files from tmp
    for extra in tmp.rglob("vlm_captions.json"):
        cap_dest = PROJECT_ROOT / "results" / "vlm_captions.json"
        cap_dest.parent.mkdir(parents=True, exist_ok=True)
        if not cap_dest.exists():
            log(f"Copying captions → {cap_dest}")
            shutil.copy2(extra, cap_dest)

    shutil.rmtree(tmp, ignore_errors=True)

    if not marker.exists():
        err("After move, MMHS150K_GT.json still missing.")
        return 1

    # Kaggle zip may ship ocr_consolidated_filtered.json; loaders expect ocr_filtered.json
    filtered = dest / "ocr_filtered.json"
    alt = dest / "ocr_consolidated_filtered.json"
    if not filtered.exists() and alt.exists():
        shutil.copy2(alt, filtered)
        ok(f"Copied {alt.name} → ocr_filtered.json for loader compatibility")

    ok(f"Dataset ready at {dest}")
    if args.skip_verify:
        return 0
    return _verify(dest)


def _verify(dest: Path) -> int:
    log("Running automated dataset verification…")
    verify = PROJECT_ROOT / "scripts" / "verify_dataset.py"
    rc = subprocess.call(
        [sys.executable, str(verify), "--dest", str(dest)],
        cwd=str(PROJECT_ROOT),
    )
    if rc == 0:
        ok("Verification passed")
    else:
        err("Verification failed — see messages above")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
