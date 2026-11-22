"""Phase 2 preprocessing: contrast normalization and tiling.

Why tiles? Shrinking a 4096-px strip to a CNN's 224 px input would divide every
length by ~18 and erase defects that are only a few pixels wide. Cutting the
strip into 256x256 tiles keeps full resolution.
"""
import cv2
import numpy as np

from . import config as C

_clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP,
                         tileGridSize=(C.STRIP_H // C.CLAHE_CELL, C.STRIP_W // C.CLAHE_CELL))


def normalize(img: np.ndarray) -> np.ndarray:
    """CLAHE on the whole strip with 32x32-pixel cells (same as 8x8 cells per 256 tile)."""
    return _clahe.apply(img)


def to_tiles(strip: np.ndarray, tile: int = C.TILE) -> np.ndarray:
    """(256, 4096) strip -> (16, 256, 256) stack of tiles, left to right."""
    h, w = strip.shape[:2]
    assert h == tile and w % tile == 0, f"unexpected strip size {strip.shape}"
    return np.stack([strip[:, i:i + tile] for i in range(0, w, tile)])


def from_tiles(tiles: np.ndarray) -> np.ndarray:
    """Inverse of to_tiles: stitch tiles back into the full strip."""
    return np.concatenate(list(tiles), axis=1)
