"""Loading AITEX images and masks, and making a leakage-safe split.

File names look like 0044_019_04.png:
    0044 = image number, 019 = defect code (000 = no defect), 04 = fabric code.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

import cv2
import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Item:
    name: str          # file name, e.g. 0044_019_04.png
    path: str          # full path to the image
    fabric: str        # fabric code, e.g. "04"
    defect: str        # defect code, "000" if defect-free
    is_defect: int     # 1 = defective, 0 = defect-free


def parse_name(name: str):
    stem = name.rsplit(".", 1)[0]
    _, defect, fabric = stem.split("_")[:3]
    return defect, fabric


def load_gray(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_strip(path) -> tuple[np.ndarray, int]:
    """Image padded to the standard 4096 width, plus the real (valid) width.

    One AITEX image (0094_027_05) is only 3796 px wide; it is padded by
    mirroring its right edge, and the padded part is ignored when scoring.
    """
    img = load_gray(path)
    w = img.shape[1]
    if w < C.STRIP_W:
        img = cv2.copyMakeBorder(img, 0, 0, 0, C.STRIP_W - w, cv2.BORDER_REFLECT)
    return img, w


def load_mask(name: str) -> np.ndarray | None:
    """Ground-truth mask as a bool array, or None if the image has no mask.

    Two AITEX images have their mask split into _mask1 and _mask2 files;
    those are merged (union) here.
    """
    stem = name.rsplit(".", 1)[0]
    files = sorted(C.MASK_DIR.glob(f"{stem}_mask*.png"))
    if not files:
        return None
    mask = np.zeros((C.STRIP_H, C.STRIP_W), bool)
    for f in files:
        m = load_gray(f) > 0
        mask[:, :m.shape[1]] |= m
    return mask


def list_items() -> list[Item]:
    items = []
    for p in sorted(C.DEFECT_DIR.glob("*.png")):
        d, f = parse_name(p.name)
        items.append(Item(p.name, str(p), f, d, 1))
    for p in sorted(C.NODEFECT_DIR.glob("*/*.png")):
        d, f = parse_name(p.name)
        items.append(Item(p.name, str(p), f, d, 0))
    return items


def make_split(items: list[Item]) -> pd.DataFrame:
    """Assign every IMAGE (not tile) to fit / calib / test / excluded.

    - fit:   defect-free images used to learn what normal fabric looks like
    - calib: defect-free images used only to choose the PASS/FAIL threshold
    - test:  the rest of the defect-free images + every defective image with a usable mask
    - excluded: defective images whose mask is missing or empty (cannot be scored)
    """
    rng = random.Random(C.SEED)
    rows = []
    by_fabric: dict[str, list[Item]] = {}
    for it in items:
        if it.is_defect:
            m = load_mask(it.name)
            usable = m is not None and m.any()
            rows.append({**it.__dict__, "split": "test" if usable else "excluded",
                         "mask_pixels": int(m.sum()) if m is not None else 0})
        else:
            by_fabric.setdefault(it.fabric, []).append(it)
    for fabric, its in sorted(by_fabric.items()):
        its = sorted(its, key=lambda x: x.name)
        rng.shuffle(its)
        for i, it in enumerate(its):
            split = "fit" if i < C.N_FIT else "calib" if i < C.N_FIT + C.N_CALIB else "test"
            rows.append({**it.__dict__, "split": split, "mask_pixels": 0})
    df = pd.DataFrame(rows)
    df["defect_name"] = df["defect"].map(C.DEFECT_NAMES).fillna("No defect")
    return df
