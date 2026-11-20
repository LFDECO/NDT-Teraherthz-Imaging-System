"""Fetch and curate 24 real industrial RGB defect image-mask pairs using pooled concurrent HTTP."""

from __future__ import annotations

import concurrent.futures
import csv
import io
import socket
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import urllib3.util.connection as urllib3_conn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Prefer IPv4 to avoid 20s Windows IPv6 fallback stalls on raw.githubusercontent.com
urllib3_conn.allowed_gai_family = lambda: socket.AF_INET

API_BASE = "https://api.github.com/repos/abin24/Magnetic-tile-defect-datasets./contents"
RAW_BASE = "https://raw.githubusercontent.com/abin24/Magnetic-tile-defect-datasets./master"

CATEGORIES = [
    ("MT_Blowhole", True),
    ("MT_Crack", True),
    ("MT_Break", True),
    ("MT_Free", False),
]


def _fetch_category_stems(session: requests.Session, cat: str) -> List[str]:
    resp = session.get(f"{API_BASE}/{cat}/Imgs", timeout=15)
    resp.raise_for_status()
    items = resp.json()
    names = {item["name"] for item in items if item.get("type") == "file"}
    stems = sorted(
        Path(n).stem
        for n in names
        if n.endswith(".jpg") and f"{Path(n).stem}.png" in names
    )
    return stems


def _download_pair(
    session: requests.Session,
    cat: str,
    stem: str,
    expect_defect: bool,
    out_dir: Path,
) -> Optional[Tuple[str, Path, Path, int, int, float]]:
    sample_id = f"{cat}__{stem}"
    local_img = out_dir / f"{sample_id}.jpg"
    local_msk = out_dir / f"{sample_id}_mask.png"

    if local_img.exists() and local_msk.exists():
        pil_img = Image.open(local_img).convert("RGB")
        pil_msk = Image.open(local_msk).convert("L")
        msk_arr = (np.asarray(pil_msk) > 127).astype(np.uint8)
        fg_pct = float(msk_arr.mean() * 100.0)
        w, h = pil_img.size
        return sample_id, local_img, local_msk, h, w, fg_pct

    img_url = f"{RAW_BASE}/{cat}/Imgs/{stem}.jpg"
    msk_url = f"{RAW_BASE}/{cat}/Imgs/{stem}.png"
    try:
        r_img = session.get(img_url, timeout=15)
        r_msk = session.get(msk_url, timeout=15)
        r_img.raise_for_status()
        r_msk.raise_for_status()
        pil_img = Image.open(io.BytesIO(r_img.content)).convert("RGB")
        pil_msk = Image.open(io.BytesIO(r_msk.content)).convert("L")
    except Exception:
        return None

    msk_arr = (np.asarray(pil_msk) > 127).astype(np.uint8)
    fg_pct = float(msk_arr.mean() * 100.0)
    fg_px = int(msk_arr.sum())
    if expect_defect and (fg_px < 12 or fg_pct > 45.0):
        return None
    if (not expect_defect) and fg_px > 0:
        return None

    pil_img.save(local_img, quality=92)
    Image.fromarray(msk_arr * 255).save(local_msk)
    w, h = pil_img.size
    return sample_id, local_img, local_msk, h, w, fg_pct


def main() -> None:
    out_dir = REPO_ROOT / "data" / "rgb_defect_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "rgb_manifest.csv"

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=12, pool_maxsize=12)
    session.mount("https://", adapter)

    existing_by_cat: Dict[str, List[str]] = {}
    existing_split: Dict[str, str] = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fp:
            for prev_row in csv.DictReader(fp):
                sid = prev_row.get("sample_id", "")
                p_cat = prev_row.get("category", "")
                p_split = prev_row.get("split", "")
                if "__" not in sid or not p_cat:
                    continue
                p_stem = sid.split("__", 1)[1]
                if (out_dir / f"{sid}.jpg").exists() and (out_dir / f"{sid}_mask.png").exists():
                    existing_by_cat.setdefault(p_cat, [])
                    if p_stem not in existing_by_cat[p_cat]:
                        existing_by_cat[p_cat].append(p_stem)
                    if p_split in {"train", "test"}:
                        existing_split[sid] = p_split

    rows: List[Dict[str, object]] = []
    target_per_cat = 6  # 4 train + 2 held-out test per category

    for cat, expect_defect in CATEGORIES:
        cached_imgs = sorted(out_dir.glob(f"{cat}__*.jpg"))
        sorted_cached_stems = [
            p.stem.split("__", 1)[1]
            for p in cached_imgs
            if (out_dir / f"{p.stem}_mask.png").exists()
        ]
        ordered_existing = existing_by_cat.get(cat, [])
        cached_stems = ordered_existing + [s for s in sorted_cached_stems if s not in ordered_existing]

        if len(cached_stems) >= target_per_cat:
            ordered_candidates = cached_stems[:target_per_cat]
        else:
            stems = _fetch_category_stems(session, cat)
            stride = max(1, len(stems) // 16)
            ordered_candidates = cached_stems + [s for s in (stems[::stride] + stems) if s not in cached_stems]

        collected: List[Tuple[str, Path, Path, int, int, float]] = []
        batch_start = 0
        while len(collected) < target_per_cat and batch_start < len(ordered_candidates):
            needed = (target_per_cat - len(collected)) + 3
            batch = ordered_candidates[batch_start : batch_start + needed]
            batch_start += len(batch)

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                futs = [
                    pool.submit(_download_pair, session, cat, st, expect_defect, out_dir)
                    for st in batch
                ]
                for fut in futs:
                    res = fut.result()
                    if res is not None and len(collected) < target_per_cat:
                        collected.append(res)

        if len(collected) < target_per_cat:
            raise RuntimeError(f"Only collected {len(collected)}/{target_per_cat} valid samples for {cat}")

        for idx, (sample_id, local_img, local_msk, h, w, fg_pct) in enumerate(collected[:target_per_cat]):
            split = existing_split.get(sample_id, "train" if idx < 4 else "test")
            rows.append(
                {
                    "sample_id": sample_id,
                    "category": cat,
                    "split": split,
                    "has_defect": int(expect_defect),
                    "image_path": local_img.relative_to(REPO_ROOT).as_posix(),
                    "mask_path": local_msk.relative_to(REPO_ROOT).as_posix(),
                    "native_height": h,
                    "native_width": w,
                    "defect_area_pct": round(fg_pct, 3),
                }
            )
            print(f"  + [{split:5s}] {sample_id:30s} | {h}x{w} | mask_fg={fg_pct:.3f}%", flush=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved {len(rows)} real industrial RGB samples & manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
