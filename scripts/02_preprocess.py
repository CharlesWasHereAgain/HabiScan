"""Phase 2 - Preprocessing + leakage-safe split.

1. Splits every IMAGE (never a tile) into fit / calib / test -> results/splits.csv
2. Checks that tiling is lossless on every image (strip -> 16 tiles -> strip)
3. Saves before/after figures for the deck and the defense

Run from the project folder:  python scripts/02_preprocess.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from habiscan import config as C, data, preprocess as P  # noqa: E402

C.FIGURES.mkdir(parents=True, exist_ok=True)
items = data.list_items()
split = data.make_split(items)
# store paths relative to the project folder so the CSV works on any computer
split.assign(path=split.path.map(lambda p: Path(p).relative_to(C.ROOT).as_posix())).to_csv(
    C.RESULTS / "splits.csv", index=False)

# --- leakage check: no image name appears in more than one split
dupes = split.groupby("name")["split"].nunique()
assert (dupes == 1).all(), "an image appears in two splits"

# --- tiling must be lossless
for it in items:
    img, _ = data.load_strip(it.path)
    assert np.array_equal(P.from_tiles(P.to_tiles(img)), img), it.name

summary = {
    "counts": split.groupby(["split", "is_defect"]).size().rename("n").reset_index().to_dict("records"),
    "tiles_per_image": C.STRIP_W // C.TILE,
    "tiling_lossless": True,
    "no_image_in_two_splits": True,
}
(C.RESULTS / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
print(split.groupby(["split", "is_defect"]).size())

# --- figure: before/after CLAHE, one tile per fabric (defect-free)
normals = split[(split.is_defect == 0) & (split.split == "fit")].groupby("fabric").first()
fig, axs = plt.subplots(2, len(normals), figsize=(2.3 * len(normals), 5), dpi=150)
for j, (fab, row) in enumerate(normals.iterrows()):
    img, _ = data.load_strip(row.path)
    t = slice(1792, 2048)                          # a tile from the middle of the strip
    axs[0, j].imshow(img[:, t], cmap="gray", vmin=0, vmax=255)
    axs[1, j].imshow(P.normalize(img)[:, t], cmap="gray", vmin=0, vmax=255)
    axs[0, j].set_title(f"Fabric {fab}", fontsize=11)
    for a in axs[:, j]:
        a.set_xticks([]); a.set_yticks([])
axs[0, 0].set_ylabel("Raw"); axs[1, 0].set_ylabel("CLAHE")
fig.tight_layout(); fig.savefig(C.FIGURES / "phase2_clahe_by_fabric.png"); plt.close(fig)
print("saved", C.FIGURES / "phase2_clahe_by_fabric.png")
