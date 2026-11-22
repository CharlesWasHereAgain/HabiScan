"""Phase 3: classical Gabor-texture anomaly detection.

Idea: woven fabric is a repeating texture. A Gabor filter responds strongly to
stripes at one angle and one spacing. We measure "texture energy" with a bank of
16 filters (8 angles x 2 scales). A defect breaks the weave, so its energy
pattern is unusual compared with normal fabric.

Three variants (each fixes a failure seen in the one before it):
  - "self":     compares each pixel with the rest of the SAME image (no training at all)
  - "fabric":   compares each pixel with defect-free images of the same fabric
                (one median and spread per filter)
  - "position": like "fabric", but separately for every 64-px column band. The
                camera is fixed, so the fabric's edge and the machine behind it
                always sit at the same place; a position-aware reference stops
                them from looking like defects. (Diagnosed on defect-free
                calibration images only.)
No weights are learned: the reference is just medians and spreads.
"""
from __future__ import annotations

import cv2
import numpy as np

THETAS = np.arange(8) * np.pi / 8          # 8 orientations
LAMBDAS = (4.0, 8.0)                        # 2 wavelengths (pixels per stripe)
ENERGY_SIGMA = 4.0                          # smoothing of squared response
BORDER = 12                                 # pixels ignored at the image edge (filter edge effects)
TOP_K = 200                                 # image score = mean of the top-K pixel scores


def _kernel(theta: float, lam: float) -> np.ndarray:
    sigma = 0.56 * lam                      # ~1-octave bandwidth
    size = int(2 * round(3 * sigma) + 1)
    k = cv2.getGaborKernel((size, size), sigma, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
    return k - k.mean()                     # zero mean: flat areas give no response


BANK = [_kernel(t, l) for l in LAMBDAS for t in THETAS]


def energy(img_u8: np.ndarray) -> np.ndarray:
    """(H, W) uint8 -> (16, H, W) float32 log texture energy."""
    x = img_u8.astype(np.float32) / 255.0
    out = np.empty((len(BANK),) + x.shape, np.float32)
    for i, k in enumerate(BANK):
        r = cv2.filter2D(x, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
        out[i] = np.log1p(cv2.GaussianBlur(r * r, (0, 0), ENERGY_SIGMA) * 100.0)
    return out


def robust_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-filter median and MAD-based spread. values: (16, N)."""
    med = np.median(values, axis=1)
    mad = np.median(np.abs(values - med[:, None]), axis=1) * 1.4826 + 1e-6
    return med, mad


def sample_pixels(e: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random sample of n interior pixels from an energy stack -> (16, n)."""
    h, w = e.shape[1:]
    ys = rng.integers(BORDER, h - BORDER, n)
    xs = rng.integers(BORDER, w - BORDER, n)
    return e[:, ys, xs]


def anomaly_map(e: np.ndarray, med: np.ndarray, mad: np.ndarray) -> np.ndarray:
    """Largest robust z-score across the 16 filters, per pixel. Border set to 0."""
    z = np.abs(e - med[:, None, None]) / mad[:, None, None]
    a = z.max(axis=0)
    a[:BORDER, :] = 0; a[-BORDER:, :] = 0; a[:, :BORDER] = 0; a[:, -BORDER:] = 0
    return a


def self_stats(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """'self' variant: statistics of the image itself (interior pixels, subsampled)."""
    inner = e[:, BORDER:-BORDER:2, BORDER:-BORDER:2].reshape(e.shape[0], -1)
    return robust_stats(inner)


def image_score(a: np.ndarray, k: int = TOP_K) -> float:
    flat = a.ravel()
    return float(np.partition(flat, -k)[-k:].mean())


BAND = 64                                   # column band width for the position-aware reference
MAD_FLOOR = 0.25                            # spread never below 25% of the fabric's typical spread
ENV_Q = 99.9                                # per-position "normal ceiling" percentile


def band_samples(e: np.ndarray, n_per_band: int, rng: np.random.Generator, valid_w: int) -> list:
    """Per column band, a random sample of interior pixels -> list of (16, n) arrays (None if band invalid)."""
    out = []
    for b in range(e.shape[2] // BAND):
        x0, x1 = max(b * BAND, BORDER), min((b + 1) * BAND, valid_w - BORDER)
        if x1 - x0 < 8:
            out.append(None); continue
        ys = rng.integers(BORDER, e.shape[1] - BORDER, n_per_band)
        xs = rng.integers(x0, x1, n_per_band)
        out.append(e[:, ys, xs])
    return out


def _to_columns(per_band: np.ndarray, width: int) -> np.ndarray:
    """(k, n_bands) values at band centres -> (k, width), linearly interpolated (no hard steps)."""
    centres = (np.arange(per_band.shape[1]) + 0.5) * BAND
    cols = np.arange(width)
    return np.stack([np.interp(cols, centres, row) for row in per_band]).astype(np.float32)


def fit_positional(samples_per_image: list, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Band samples from defect-free images -> per-column median and spread, each (16, width)."""
    n_bands = len(samples_per_image[0])
    med = np.zeros((len(BANK), n_bands)); mad = np.ones_like(med)
    for b in range(n_bands):
        vals = [s[b] for s in samples_per_image if s[b] is not None]
        if vals:
            med[:, b], mad[:, b] = robust_stats(np.concatenate(vals, axis=1))
    typical = np.median(mad, axis=1, keepdims=True)
    mad = np.maximum(mad, MAD_FLOOR * typical)                       # flat regions can't explode
    return _to_columns(med, width), _to_columns(mad, width)


def anomaly_map_positional(e: np.ndarray, medc: np.ndarray, madc: np.ndarray) -> np.ndarray:
    """Largest robust z-score across filters, using the reference for each column."""
    a = (np.abs(e - medc[:, None, :]) / madc[:, None, :]).max(axis=0)
    a[:BORDER, :] = 0; a[-BORDER:, :] = 0; a[:, :BORDER] = 0; a[:, -BORDER:] = 0
    return a


def fit_envelope(maps_and_widths: list, width: int) -> np.ndarray:
    """How high a NORMAL score gets at each position: per band, the 99.9th percentile of
    defect-free anomaly maps (median over images) -> per-column ceiling (width,)."""
    n_bands = width // BAND
    per_img = np.full((len(maps_and_widths), n_bands), np.nan)
    for i, (a, w) in enumerate(maps_and_widths):
        for b in range(n_bands):
            x0, x1 = max(b * BAND, BORDER), min((b + 1) * BAND, w - BORDER)
            if x1 - x0 >= 8:
                per_img[i, b] = np.percentile(a[BORDER:-BORDER, x0:x1], ENV_Q)
    env = np.nanmedian(per_img, axis=0)
    env = np.where(np.isnan(env), np.nanmedian(env), env)
    env = np.maximum(env, np.median(env))                            # never more sensitive than typical fabric
    return _to_columns(env[None, :], width)[0]
