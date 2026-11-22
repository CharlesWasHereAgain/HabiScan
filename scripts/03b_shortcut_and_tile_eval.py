"""Phase 3b - Is the detector seeing defects, or a shortcut?

Finding: in AITEX, several fabrics' defective images were captured brighter,
with more contrast or sharper than their defect-free images. An image-level
detector can tell the two GROUPS apart without ever looking at the defect.

Two checks:
  1. Erase test: zero out the defect region (dilated 32 px) and re-score.
     If defective images are still flagged, the image-level score is not
     driven by the defect.
  2. Tile-level test (shortcut-proof): within each defective image, compare
     the tiles that contain the defect with the tiles that don't. Same image,
     same lighting, same session -> only the defect can make a tile stand out.
     Metrics: tile AUROC, and how often the highest-scoring tile contains the
     defect (vs. the chance rate of picking it at random).

Run from the project folder (after 03_gabor_baseline.py):
    python scripts/03b_shortcut_and_tile_eval.py
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from habiscan import config as C, data, gabor as G, metrics as M  # noqa: E402

spec = importlib.util.spec_from_file_location("phase3", ROOT / "scripts" / "03_gabor_baseline.py")
p3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p3)

TILE_TOPK = 50
ERASE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))     # 32-px dilation
scores = pd.read_csv(C.RESULTS / "phase3_image_scores.csv", dtype={"fabric": str})
names = p3.split.set_index("name")
test_defects = names[(names.split == "test") & (names.is_defect == 1)]


def tile_scores(a, w):
    out = []
    for i in range(C.STRIP_W // C.TILE):
        x0, x1 = i * C.TILE + G.BORDER, min((i + 1) * C.TILE, w) - G.BORDER
        if x1 - x0 < 32:
            out.append(np.nan); continue
        t = a[G.BORDER:-G.BORDER, x0:x1].ravel()
        out.append(float(np.partition(t, -TILE_TOPK)[-TILE_TOPK:].mean()))
    return np.array(out)


results, tile_rows = {}, []
for setting, variant in p3.CONFIGS:
    ref = p3.fit_reference(setting)
    g = scores[(scores.setting == setting) & (scores.variant == variant)]
    thr = float(np.quantile(g[g.split == "calib"].score, p3.IMG_Q))
    erased, hits, chance, t_lab, t_sc = [], [], [], [], []
    for name, r in test_defects.iterrows():
        img, w = data.load_strip(r.abs_path)
        gt = data.load_mask(name)
        a = p3.anomaly(img, w, setting, variant, ref, r.fabric)
        a_er = a.copy(); a_er[cv2.dilate(gt.astype(np.uint8), ERASE) > 0] = 0
        erased.append(G.image_score(a_er))
        ts = tile_scores(a, w)
        lab = np.array([gt[:, i * C.TILE:(i + 1) * C.TILE].any() for i in range(C.STRIP_W // C.TILE)])
        ok = ~np.isnan(ts)
        ts, lab = ts[ok], lab[ok]
        hits.append(bool(lab[np.argmax(ts)])); chance.append(lab.mean())
        t_lab.extend(lab.astype(int)); t_sc.extend(ts)
        if (setting, variant) == p3.MAIN:
            tile_rows.append({"name": name, "defect_name": r.defect_name, "fabric": r.fabric,
                              "top_tile_has_defect": hits[-1], "defect_tiles": int(lab.sum()), "tiles": int(lab.size)})
    normal_test = g[(g.split == "test") & (g.is_defect == 0)].score.to_numpy()
    y = np.r_[np.zeros(normal_test.size), np.ones(len(erased))]
    results[f"{setting}+{variant}"] = {
        "image_auroc_original": float(M.image_metrics(g[g.split == "test"].is_defect, g[g.split == "test"].score, thr)["auroc"]),
        "image_auroc_with_defects_erased": float(M.image_metrics(y, np.r_[normal_test, erased], thr)["auroc"]),
        "defect_images_still_flagged_after_erase": int((np.array(erased) > thr).sum()),
        "tile_auroc_within_defect_images": float(M.pixel_auroc(np.array(t_lab), np.array(t_sc))),
        "top_tile_hit_rate": float(np.mean(hits)),
        "top_tile_chance_rate": float(np.mean(chance)),
        "n_defect_images": len(hits),
    }
    print(setting, variant, json.dumps(results[f"{setting}+{variant}"], indent=1))

(C.RESULTS / "phase3b_shortcut_and_tile_eval.json").write_text(json.dumps(results, indent=2))
tr = pd.DataFrame(tile_rows); tr.to_csv(C.RESULTS / "phase3b_tile_hits_main.csv", index=False)
print(tr.groupby("defect_name").top_tile_has_defect.agg(["size", "sum", "mean"]).sort_values("size", ascending=False))

# figure: original vs erased image AUROC, and top-tile hit vs chance (main + no-reference)
BLUE, GRAY, GRID = "#2a78d6", "#9aa3b2", "#e4e3df"
keys = [("clahe+self", "No reference"), ("clahe+fabric", "Per-fabric"), ("clahe+position", "Position-aware")]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
x = np.arange(len(keys)); wbar = 0.38
axs[0].bar(x - wbar / 2, [results[k]["image_auroc_original"] for k, _ in keys], wbar, color=BLUE, label="Original")
axs[0].bar(x + wbar / 2, [results[k]["image_auroc_with_defects_erased"] for k, _ in keys], wbar, color=GRAY, label="Defect region erased")
axs[0].axhline(0.5, color="#52514e", lw=1, ls="--"); axs[0].set_ylim(0, 1)
axs[0].set_xticks(x, [l for _, l in keys]); axs[0].set_ylabel("Image AUROC"); axs[0].legend(frameon=False, loc="upper left")
axs[0].set_title("Image-level score: barely changes when the defect is erased", fontsize=12)
axs[1].bar(x - wbar / 2, [results[k]["top_tile_hit_rate"] * 100 for k, _ in keys], wbar, color=BLUE, label="Top tile contains the defect")
axs[1].bar(x + wbar / 2, [results[k]["top_tile_chance_rate"] * 100 for k, _ in keys], wbar, color=GRAY, label="Chance (random tile)")
axs[1].set_xticks(x, [l for _, l in keys]); axs[1].set_ylabel("% of defect images"); axs[1].legend(frameon=False, loc="upper left")
axs[1].set_ylim(0, 100); axs[1].set_title("Tile-level test (same image, same lighting)", fontsize=12)
for ax in axs:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(C.FIGURES / "phase3b_shortcut_check.png"); plt.close(fig)
print("saved", C.FIGURES / "phase3b_shortcut_check.png")

# figure: localization examples for the MAIN configuration (full strips, top tile boxed)
ref = p3.fit_reference(p3.MAIN[0])
hit = tr[tr.top_tile_has_defect].head(3).assign(kind="Top tile = defect")
miss = tr[~tr.top_tile_has_defect].sample(3, random_state=C.SEED).assign(kind="Top tile elsewhere")
fig, axs = plt.subplots(6, 1, figsize=(14, 6.6), dpi=150)
for ax, (_, r) in zip(axs, pd.concat([hit, miss]).iterrows()):
    row = names.loc[r["name"]]
    img, w = data.load_strip(row.abs_path); gt = data.load_mask(r["name"])
    a = p3.anomaly(img, w, *p3.MAIN, ref, row.fabric)
    ts = tile_scores(a, w); best = int(np.nanargmax(ts))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax.imshow(np.clip(a, 0, 3), cmap="inferno", alpha=0.45, vmin=0, vmax=3, aspect="auto")
    ax.contour(gt, levels=[0.5], colors=["#36d6c3"], linewidths=1.5)
    ax.add_patch(plt.Rectangle((best * C.TILE, 0), C.TILE - 1, C.STRIP_H - 1, fill=False, ec="#ffffff", lw=2))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(f"{r['kind']}\n{r.defect_name}", rotation=0, ha="right", va="center", fontsize=10)
fig.subplots_adjust(left=0.16, right=0.995, top=0.995, bottom=0.005, hspace=0.08)
fig.savefig(C.FIGURES / "phase3b_localization_examples.png"); plt.close(fig)
print("saved", C.FIGURES / "phase3b_localization_examples.png")
