"""Phase 4 - PatchCore: "memorize normal fabric, flag whatever doesn't match".

PatchCore (Roth et al., CVPR 2022) needs NO defect images for training:
  1. A ResNet-18 pretrained on ImageNet (frozen, never retrained) turns every
     256x256 tile into a 32x32 grid of patch descriptors (384 numbers each,
     from layer2 + layer3, each averaged over its 3x3 neighbourhood).
  2. Descriptors from the defect-free FIT images of one fabric go into that
     fabric's memory bank. Greedy coreset sampling keeps 1% of them, chosen to
     cover the whole bank (so rare-but-normal patches are not thrown away).
     The paper shows 1% loses almost nothing vs. 10% and is ~10x faster on a CPU.
  3. A new patch's anomaly score = distance to its nearest neighbour in the
     bank. Far from every normal patch = probably a defect.

Same input and output as the Gabor baseline: a CLAHE strip goes in, a
(256, 4096) anomaly map comes out, so both are scored with identical code.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import config as C, preprocess as P

WEIGHTS_URL = ("https://github.com/huggingface/pytorch-image-models/releases/download/"
               "v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth")   # timm ResNet-18, ImageNet-1k
WEIGHTS_FILE = C.ROOT / "models" / "resnet18_a1_0-d63eafa0.pth"

GRID = C.TILE // 8              # 32x32 patch grid per tile (layer2 has stride 8)
CORESET_RATIO = 0.01            # keep 1% of each fabric's normal patches (~1,300 per fabric)
PROJ_DIM = 128                  # random projection used only to SPEED UP coreset selection
SMOOTH_SIGMA = 4.0              # Gaussian smoothing of the upsampled map (as in the paper)
BORDER = 12                     # same ignored border as the Gabor baseline
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

torch.set_grad_enabled(False)


def load_backbone():
    """Frozen ResNet-18 returning layer2 (stride 8) and layer3 (stride 16) feature maps."""
    import timm
    model = timm.create_model("resnet18", pretrained=False, features_only=True, out_indices=(2, 3))
    if not WEIGHTS_FILE.exists():
        WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, str(WEIGHTS_FILE))
    state = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=True)
    missing, _ = model.load_state_dict(state, strict=False)       # fc.* weights are unused
    assert not missing, f"backbone weights missing: {missing[:3]}"
    return model.eval()


def features(model, strip_u8: np.ndarray) -> torch.Tensor:
    """CLAHE strip (256, 4096) -> (16 tiles, 32, 32, 384) patch descriptors."""
    tiles = P.to_tiles(P.normalize(strip_u8)).astype(np.float32) / 255.0
    x = torch.from_numpy(tiles)[:, None].repeat(1, 3, 1, 1)          # gray -> 3 channels
    x = (x - MEAN) / STD
    f2, f3 = model(x)
    f2 = F.avg_pool2d(f2, 3, 1, 1)                                     # local neighbourhood
    f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([f2, f3], 1).permute(0, 2, 3, 1).contiguous()    # (16, 32, 32, 384)


def valid_tiles(w: int) -> int:
    """Number of tiles that hold real fabric (the one narrow image has mirrored padding)."""
    return int(np.ceil(w / C.TILE))


def greedy_coreset(x: torch.Tensor, ratio: float = CORESET_RATIO, seed: int = C.SEED) -> torch.Tensor:
    """Pick ratio*N rows of x that cover x as evenly as possible (k-center greedy)."""
    n = x.shape[0]; k = max(1, int(n * ratio))
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(x.shape[1], PROJ_DIM, generator=g) / PROJ_DIM ** 0.5
    z = x @ proj
    zz = (z * z).sum(1)                                   # squared norms, reused every step
    i = int(torch.randint(n, (1,), generator=g))
    picked = [i]
    dist = zz - 2 * z @ z[i] + zz[i]                      # squared distance to the picked set
    for _ in range(k - 1):
        i = int(torch.argmax(dist))
        picked.append(i)
        dist = torch.minimum(dist, zz - 2 * z @ z[i] + zz[i])
    return x[torch.tensor(picked)]


def nearest_distance(q: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """For each query row, Euclidean distance to its nearest neighbour in the bank."""
    out = torch.empty(q.shape[0])
    for i in range(0, q.shape[0], chunk):
        out[i:i + chunk] = torch.cdist(q[i:i + chunk], bank).min(1).values
    return out


def patch_map(feats: torch.Tensor, bank: torch.Tensor, w: int) -> np.ndarray:
    """Patch descriptors (16, 32, 32, 384) -> (32, 512) patch-level anomaly map."""
    t = feats.shape[0]
    d = nearest_distance(feats.reshape(-1, feats.shape[-1]), bank).reshape(t, GRID, GRID).numpy()
    d[valid_tiles(w):] = 0
    return np.concatenate(list(d), axis=1)


def pixel_map(pm: np.ndarray, w: int) -> np.ndarray:
    """(32, 512) patch map -> (256, 4096) pixel map: upsample, smooth, zero border/padding."""
    a = cv2.resize(pm.astype(np.float32), (C.STRIP_W, C.STRIP_H), interpolation=cv2.INTER_LINEAR)
    a = cv2.GaussianBlur(a, (0, 0), SMOOTH_SIGMA)
    a[:BORDER, :] = 0; a[-BORDER:, :] = 0; a[:, :BORDER] = 0
    a[:, max(w - BORDER, BORDER):] = 0
    return a


def fit_bank(model, strips: list[tuple[np.ndarray, int]]) -> torch.Tensor:
    """Defect-free strips of ONE fabric -> coreset memory bank (M, 384)."""
    feats = []
    for img, w in strips:
        f = features(model, img)[:valid_tiles(w)]
        feats.append(f.reshape(-1, f.shape[-1]))
    return greedy_coreset(torch.cat(feats))


ENV_BAND = 8                    # 8 patch columns = 64 px, same band width as the Gabor reference
ENV_Q = 99.0                    # per-position "normal ceiling" percentile


def fit_envelope(model, strips: list[tuple[np.ndarray, int]]) -> np.ndarray:
    """How high a NORMAL patch score gets at each position across the strip -> (512,).

    Why: on defect-free calibration images, the hottest tile of plain PatchCore sits
    at the left edge of the strip (the fixed camera sees the selvage and machine
    there), exactly like the first Gabor versions. Fix, using defect-free FIT images
    only: split them in two halves, build a bank from one half, score the other
    (and vice versa), and record per column band the 99th-percentile score.
    The final map is divided by this ceiling, so 1.0 = "as high as normal gets here".
    """
    half = len(strips) // 2
    groups = (strips[:half], strips[half:])
    per_img = []
    for a, b in ((0, 1), (1, 0)):
        bank = fit_bank(model, groups[a])
        for img, w in groups[b]:
            pm = patch_map(features(model, img), bank, w)
            n_bands = pm.shape[1] // ENV_BAND
            valid = valid_tiles(w) * GRID
            per_img.append([np.percentile(pm[:, i * ENV_BAND:(i + 1) * ENV_BAND], ENV_Q)
                            if (i + 1) * ENV_BAND <= valid else np.nan for i in range(n_bands)])
    env = np.nanmedian(np.array(per_img), axis=0)
    env = np.where(np.isnan(env), np.nanmedian(env), env)
    centres = (np.arange(env.size) + 0.5) * ENV_BAND
    return np.interp(np.arange(C.STRIP_W // 8), centres, env).astype(np.float32)


def save_banks(banks: dict, path: Path):
    torch.save({k: v.half() for k, v in banks.items()}, path)


def load_banks(path: Path) -> dict:
    return {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
