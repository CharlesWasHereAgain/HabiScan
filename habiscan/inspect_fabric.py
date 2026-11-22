"""Phase 6 - one-image inspection with the Phase 4 model (used by app.py).

inspect(image) -> fabric used, anomaly score, PASS/FAIL, heatmap, defect area %.
Everything here is reused, unchanged, from Phase 4:
  model      frozen ImageNet ResNet-18 (habiscan/patchcore.py)
  banks      models/patchcore_banks.pt            (one per fabric, from FIT images)
  ceilings   models/patchcore_position_ceiling.npz
  thresholds results/phase4_patchcore_metrics.json (from CALIBRATION images only)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from . import config as C, gabor as G, patchcore as PC

FABRICS = ["00", "01", "02", "03", "04", "05", "06"]


@dataclass
class Result:
    fabric: str              # fabric code whose memory bank was used
    auto: bool               # True if the fabric was auto-detected
    fit_distance: dict       # median patch distance to each fabric's bank (lower = closer)
    score: float             # image anomaly score (1.0 = as high as normal fabric gets)
    pixel_map: np.ndarray    # (256, 4096) anomaly map, already divided by the position ceiling
    strip: np.ndarray        # (256, 4096) grayscale input as scored
    width: int               # real width before padding
    seconds: float


class Inspector:
    def __init__(self):
        self.model = PC.load_backbone()
        self.banks = PC.load_banks(C.ROOT / "models" / "patchcore_banks.pt")
        self.ceil = dict(np.load(C.ROOT / "models" / "patchcore_position_ceiling.npz"))
        m = json.loads((C.RESULTS / "phase4_patchcore_metrics.json").read_text())["main"]
        self.image_threshold = float(m["threshold"])      # 95th pct of calibration image scores
        self.pixel_threshold = float(m["pixel_threshold"])  # 99.9th pct of calibration pixels

    @staticmethod
    def to_strip(img: np.ndarray) -> tuple[np.ndarray, int]:
        """Any image -> AITEX-style (256, 4096) grayscale strip + real width.

        Grayscale; scaled so the height is 256 px (aspect kept); wider than 4096
        is cut at 4096; narrower is mirror-padded (padding is ignored, as in Phase 1).
        """
        if img.ndim == 3:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
        img = img.astype(np.uint8)
        if img.shape[0] != C.STRIP_H:
            s = C.STRIP_H / img.shape[0]
            img = cv2.resize(img, (max(1, round(img.shape[1] * s)), C.STRIP_H), interpolation=cv2.INTER_AREA)
        img = img[:, :C.STRIP_W]
        w = img.shape[1]
        if w < C.STRIP_W and C.STRIP_W - w < w:          # same padding as data.load_strip
            img = cv2.copyMakeBorder(img, 0, 0, 0, C.STRIP_W - w, cv2.BORDER_REFLECT)
        elif w < C.STRIP_W:                                # very short input: repeat it mirrored
            reps = int(np.ceil(C.STRIP_W / w))
            img = np.concatenate([img, *([img[:, ::-1], img] * reps)], axis=1)[:, :C.STRIP_W]
        return img, w

    @torch.no_grad()          # web servers run requests in worker threads: switch gradients off here too
    def inspect(self, img: np.ndarray, fabric: str = "auto") -> Result:
        import time
        t0 = time.time()
        strip, w = self.to_strip(img)
        feats = PC.features(self.model, strip)
        auto = fabric in (None, "", "auto")
        fit = {}
        if auto:
            n = PC.valid_tiles(w)
            q = feats[:n].reshape(-1, feats.shape[-1])[::4]           # every 4th patch is enough
            for f in FABRICS:
                fit[f] = float(PC.nearest_distance(q, self.banks[f]).median())
            fabric = min(fit, key=fit.get)
        pm = PC.patch_map(feats, self.banks[fabric], w) / self.ceil[fabric][None, :]
        a = PC.pixel_map(pm, w)
        return Result(fabric, auto, fit, G.image_score(a), a, strip, w, time.time() - t0)

    def verdict(self, r: Result, image_threshold: float | None = None) -> dict:
        thr = self.image_threshold if image_threshold is None else image_threshold
        inner = np.zeros_like(r.pixel_map, bool)
        inner[PC.BORDER:-PC.BORDER, PC.BORDER:max(r.width - PC.BORDER, PC.BORDER + 1)] = True
        hot = (r.pixel_map > self.pixel_threshold) & inner
        tiles = [r.pixel_map[:, i * C.TILE:(i + 1) * C.TILE].max() for i in range(PC.valid_tiles(r.width))]
        return {"fail": r.score > thr, "threshold": thr, "score": r.score,
                "defect_area_pct": 100.0 * hot.sum() / inner.sum(), "hottest_tile": int(np.argmax(tiles)),
                "hot_mask": hot}

    def overlay(self, r: Result, v: dict) -> np.ndarray:
        """RGB picture: fabric + heatmap (only where it exceeds normal) + hottest tile box."""
        base = cv2.cvtColor(r.strip, cv2.COLOR_GRAY2RGB).astype(np.float32)
        heat = np.clip((r.pixel_map / self.pixel_threshold - 0.9) / 0.6, 0, 1)   # 0 below ~normal
        cmap = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32)
        alpha = (0.65 * heat)[..., None]
        out = (base * (1 - alpha) + cmap * alpha).astype(np.uint8)
        if v["fail"]:
            x0 = v["hottest_tile"] * C.TILE
            cv2.rectangle(out, (x0 + 2, 2), (x0 + C.TILE - 3, C.STRIP_H - 3), (54, 214, 195), 4)
        return out[:, :r.width]
