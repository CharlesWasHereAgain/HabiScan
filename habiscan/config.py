"""Project-wide settings for HabiScan.

Every path is built from the project folder, so the code runs the same on
Windows (VS Code) and Linux without editing anything.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../HabiScan
DATA = ROOT / "data" / "aitex"
DEFECT_DIR = DATA / "Defect_images"
MASK_DIR = DATA / "Mask_images"
NODEFECT_DIR = DATA / "NODefect_images"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# Image geometry (AITEX strips are 4096 wide x 256 tall, grayscale)
STRIP_W, STRIP_H = 4096, 256
TILE = 256                       # square tile size used by the pipeline

# Contrast normalization (CLAHE). A 32x32-pixel cell = an 8x8 grid per 256 tile.
CLAHE_CLIP = 2.0
CLAHE_CELL = 32

# Split of defect-free images per fabric (by IMAGE, never by tile)
N_FIT, N_CALIB = 8, 4            # the rest of each fabric's defect-free images are test
SEED = 106                       # fixed so the split is reproducible

# Defect type codes used in AITEX file names (nnnn_DDD_ff.png)
DEFECT_NAMES = {
    "002": "Broken end", "006": "Broken yarn", "010": "Broken pick", "016": "Weft curling",
    "019": "Fuzzyball", "022": "Cut selvage", "023": "Crease", "025": "Warp ball",
    "027": "Knots", "029": "Contamination", "030": "Nep", "036": "Weft crack",
}
