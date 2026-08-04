#!/usr/bin/env python3
"""Download all CREMA-D AudioWAV files via GitHub Media LFS mirror.

Skips files that already exist and are non-empty.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = (
    "https://media.githubusercontent.com/media/"
    "CheyneyComputerScience/CREMA-D/master/AudioWAV/"
)


def download_one(fname: str, out_dir: Path, session: requests.Session,
                 max_retries: int = 5) -> tuple[str, str]:
    dst = out_dir / fname
    if dst.exists() and dst.stat().st_size > 1024:  # >1KB heuristic
        return fname, "skip"
    url = BASE_URL + fname
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=60, stream=True)
            r.raise_for_status()
            tmp = dst.with_suffix(dst.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 15):
                    if chunk:
                        f.write(chunk)
            tmp.rename(dst)
            return fname, "ok"
        except Exception as e:
            if attempt == max_retries - 1:
                return fname, f"fail: {e!r}"
            time.sleep(2 ** attempt)
    return fname, "fail: retries"


def load_filelist(cremad_root: Path) -> list[str]:
    """Read SentenceFilenames.csv and expand to all 6 emotions/intensities."""
    # We use SentenceFilenames.csv which lists 91 speakers x 12 sentences.
    # But the actual audio filenames follow the pattern
    #     <ActorID>_<SentenceCode>_<Emotion>_<Intensity>.wav
    # SentenceFilenames.csv is not directly the audio list.
    # Instead, use processedResults/summaryTable.csv which lists every wav
    # (7442 total) via the FileName column.
    st = cremad_root / "processedResults" / "summaryTable.csv"
    if not st.exists():
        print(f"ERROR: {st} not found. Did the repo clone finish?")
        sys.exit(1)
    import csv
    names = []
    with open(st, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = row.get("FileName") or row.get("filename") or row.get("File")
            if fn:
                names.append(fn.strip() + ".wav")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cremad_root", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cremad_root = Path(args.cremad_root)
    out_dir = cremad_root / "AudioWAV"
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = load_filelist(cremad_root)
    print(f"Total files to fetch: {len(filenames)}")

    session = requests.Session()
    session.headers.update({"User-Agent": "cremad-fetcher/1.0"})

    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(download_one, fn, out_dir, session) for fn in filenames]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            fn, status = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                print(f"[{i}] {fn}: {status}", flush=True)
            if i % 200 == 0:
                dt = time.time() - t0
                print(f"[{i}/{len(filenames)}] ok={n_ok} skip={n_skip} "
                      f"fail={n_fail}  {dt:.1f}s elapsed", flush=True)

    dt = time.time() - t0
    print(f"\nDONE: ok={n_ok} skip={n_skip} fail={n_fail}  total time {dt:.1f}s")
    if n_fail > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
