"""Phase 5 - Evaluation: does HabiScan work on a fabric it has never seen?

Leave-one-fabric-out (LOFO), zero-shot. For each of the 7 fabrics F:
  - memory bank   = the Phase 4 coresets of the OTHER 6 fabrics, stacked
                    (each was built from defect-free FIT images only)
  - position ceiling = element-wise median of the other 6 fabrics' ceilings
  - image / pixel threshold = 95th / 99.9th percentile of the other 6 fabrics'
                    defect-free CALIBRATION images scored with this fold's model
  - test          = fabric F's test images (the same ones Phase 4 used)
Nothing from fabric F is used: no images, no labels, no thresholds.

Every result is shown next to Phase 4 ("in-fabric"), where F had its own 12
defect-free images (8 to build the bank, 4 to set the threshold). The gap is
what it costs to skip setting up a new fabric.

Also writes the figures needed for the "Competitive?" row: detection rate
(= recall), accuracy and false-alarm rate at the image level.

Run after Phase 4:   python scripts/05_leave_one_fabric_out.py   (~8 min on CPU)
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from habiscan import config as C, data, gabor as G, metrics as M, patchcore as PC  # noqa: E402

IMG_Q, TILE_TOPK = 0.95, 50
rng = np.random.default_rng(C.SEED)
split = pd.read_csv(C.RESULTS / "splits.csv", dtype={"fabric": str, "defect": str})
split["abs_path"] = split.path.map(lambda p: str(C.ROOT / p))
banks = PC.load_banks(C.ROOT / "models" / "patchcore_banks.pt")
envs = dict(np.load(C.ROOT / "models" / "patchcore_position_ceiling.npz"))
FABRICS = sorted(banks)


def tile_scores(a, w):
    """Score of each 256x256 tile = mean of its top-50 pixels (identical to 03b / 04)."""
    out = []
    for i in range(C.STRIP_W // C.TILE):
        x0, x1 = i * C.TILE + PC.BORDER, min((i + 1) * C.TILE, w) - PC.BORDER
        if x1 - x0 < 32:
            out.append(np.nan); continue
        t = a[PC.BORDER:-PC.BORDER, x0:x1].ravel()
        out.append(float(np.partition(t, -TILE_TOPK)[-TILE_TOPK:].mean()))
    return np.array(out)


def tile_eval(a, w, name):
    gt = data.load_mask(name)
    ts = tile_scores(a, w)
    lab = np.array([gt[:, i * C.TILE:(i + 1) * C.TILE].any() for i in range(C.STRIP_W // C.TILE)])
    ok = ~np.isnan(ts)
    return ts[ok], lab[ok]


def summarise(df, tiles):
    """Image metrics at each fold's own threshold + tile-level test."""
    y, s, pred = df.is_defect.to_numpy(), df.score.to_numpy(), df.flagged.to_numpy()
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    out = {"n_defect": int(y.sum()), "n_good": int((y == 0).sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "detection_rate": tp / max(tp + fn, 1), "false_alarm_rate": fp / max(fp + tn, 1),
           "accuracy": (tp + tn) / len(y), "precision": tp / max(tp + fp, 1)}
    out["f1"] = 2 * out["precision"] * out["detection_rate"] / max(out["precision"] + out["detection_rate"], 1e-9)
    out["image_auroc"] = M.image_metrics(y, s, 0)["auroc"] if len(set(y)) == 2 else float("nan")
    if tiles:
        lab = np.concatenate([t[1] for t in tiles]); sc = np.concatenate([t[0] for t in tiles])
        out["tile_auroc"] = M.pixel_auroc(lab.astype(int), sc)
        out["top_tile_hit_rate"] = float(np.mean([t[1][np.argmax(t[0])] for t in tiles]))
        out["top_tile_chance_rate"] = float(np.mean([t[1].mean() for t in tiles]))
    return out


if __name__ == "__main__":
    t0 = time.time()
    model = PC.load_backbone()
    fold_bank = {f: torch.cat([banks[g] for g in FABRICS if g != f]) for f in FABRICS}
    fold_env = {f: np.median(np.stack([envs[g] for g in FABRICS if g != f]), 0) for f in FABRICS}

    # ---------------- 1. score: each test image under its own LOFO fold; each calib
    #                     image under the 6 folds that hold its fabric OUT of "new"
    test_rows, calib_rows, lofo_tiles, in_tiles = [], [], {f: [] for f in FABRICS}, {f: [] for f in FABRICS}
    p4maps = np.load(C.RESULTS / "phase4_patch_maps.npz")
    todo = split[split.split.isin(["calib", "test"])]
    for k, (_, r) in enumerate(todo.iterrows()):
        img, w = data.load_strip(r.abs_path)
        feats = PC.features(model, img)
        if r.split == "calib":
            for f in FABRICS:
                if f == r.fabric:
                    continue                                     # never use the held-out fabric's images
                a = PC.pixel_map(PC.patch_map(feats, fold_bank[f], w) / fold_env[f][None, :], w)
                calib_rows.append({"fold": f, "name": r["name"], "score": G.image_score(a)})
        else:
            t1 = time.time()
            a = PC.pixel_map(PC.patch_map(feats, fold_bank[r.fabric], w) / fold_env[r.fabric][None, :], w)
            test_rows.append({"fabric": r.fabric, "name": r["name"], "is_defect": r.is_defect,
                              "defect_name": r.defect_name, "score": G.image_score(a), "seconds": time.time() - t1})
            if r.is_defect:
                lofo_tiles[r.fabric].append(tile_eval(a, w, r["name"]))
                a_in = PC.pixel_map(p4maps[r["name"]].astype(np.float32) / envs[r.fabric][None, :], w)
                in_tiles[r.fabric].append(tile_eval(a_in, w, r["name"]))
        if k % 40 == 0:
            print(f"{k}/{len(todo)} images, {time.time() - t0:.0f}s")

    calib = pd.DataFrame(calib_rows)
    thr = {f: float(np.quantile(calib[calib.fold == f].score, IMG_Q)) for f in FABRICS}
    lofo = pd.DataFrame(test_rows)
    lofo["flagged"] = [int(s > thr[f]) for s, f in zip(lofo.score, lofo.fabric)]
    lofo.to_csv(C.RESULTS / "phase5_lofo_image_scores.csv", index=False)

    p4 = pd.read_csv(C.RESULTS / "phase4_image_scores_test_position.csv", dtype={"fabric": str})
    p4_thr = json.loads((C.RESULTS / "phase4_patchcore_metrics.json").read_text())["main"]["threshold"]
    p4["flagged"] = (p4.score > p4_thr).astype(int)

    # ---------------- 2. per-fabric and pooled metrics, in-fabric vs zero-shot
    rows = []
    for f in FABRICS:
        rows.append({"fabric": f, "setting": "in-fabric (Phase 4)", **summarise(p4[p4.fabric == f], in_tiles[f])})
        rows.append({"fabric": f, "setting": "new fabric, zero-shot (LOFO)",
                     "image_threshold": thr[f], **summarise(lofo[lofo.fabric == f], lofo_tiles[f])})
    per_fabric = pd.DataFrame(rows)
    per_fabric.to_csv(C.RESULTS / "phase5_lofo_per_fabric.csv", index=False)

    pooled = {"in-fabric (Phase 4)": summarise(p4, sum(in_tiles.values(), [])),
              "new fabric, zero-shot (LOFO)": summarise(lofo, sum(lofo_tiles.values(), []))}
    # image AUROC across folds is not one ranking (each fold has its own model), so
    # report the mean of per-fabric AUROCs over fabrics with >= 5 defective test images
    big = [f for f in FABRICS if (lofo[lofo.fabric == f].is_defect.sum() >= 5)]
    for setting in pooled:
        pf = per_fabric[(per_fabric.setting == setting) & per_fabric.fabric.isin(big)]
        pooled[setting]["mean_image_auroc_per_fabric"] = float(pf.image_auroc.mean())

    # bootstrap 95% CIs over images for the pooled zero-shot numbers
    boot = {"detection_rate": [], "false_alarm_rate": [], "accuracy": []}
    for _ in range(2000):
        s = lofo.iloc[rng.integers(0, len(lofo), len(lofo))]
        m = summarise(s, [])
        for k in boot:
            boot[k].append(m[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

    # paired bootstrap over defective images: is zero-shot localization really different?
    names_def = [n for f in FABRICS for n in lofo[(lofo.fabric == f) & (lofo.is_defect == 1)].name]
    T_in = sum((in_tiles[f] for f in FABRICS), []); T_lo = sum((lofo_tiles[f] for f in FABRICS), [])
    pd.DataFrame([{"name": n, "in_fabric_top_tile_hit": bool(a[1][np.argmax(a[0])]),
                   "zero_shot_top_tile_hit": bool(b[1][np.argmax(b[0])])} for n, a, b in zip(names_def, T_in, T_lo)]
                 ).to_csv(C.RESULTS / "phase5_lofo_tile_hits.csv", index=False)
    d_auc, d_hit = [], []
    for _ in range(1000):
        pick = rng.integers(0, len(T_in), len(T_in))
        A = [T_in[i] for i in pick]; B = [T_lo[i] for i in pick]
        d_auc.append(summarise(lofo.iloc[:2], B)["tile_auroc"] - summarise(lofo.iloc[:2], A)["tile_auroc"])
        d_hit.append(np.mean([b[1][np.argmax(b[0])] for b in B]) - np.mean([a[1][np.argmax(a[0])] for a in A]))
    ci["tile_auroc_zero_shot_minus_in_fabric"] = [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))]
    ci["top_tile_hit_zero_shot_minus_in_fabric"] = [float(np.percentile(d_hit, 2.5)), float(np.percentile(d_hit, 97.5))]

    out = {"protocol": "zero-shot leave-one-fabric-out: bank = other 6 fabrics' Phase 4 coresets; "
                       "position ceiling = median of their ceilings; image threshold = 95th percentile of "
                       "their defect-free calibration scores. Nothing from the held-out fabric is used.",
           "fabrics_in_mean_auroc": big, "pooled": pooled, "zero_shot_bootstrap_95ci": ci,
           "fold_thresholds": thr, "fold_bank_sizes": {f: int(fold_bank[f].shape[0]) for f in FABRICS},
           "seconds_per_strip_zero_shot": float(lofo.seconds.mean()),
           "runtime_seconds_total": time.time() - t0}
    (C.RESULTS / "phase5_lofo_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    pd.set_option("display.width", 200)
    print(per_fabric[["fabric", "setting", "n_defect", "n_good", "image_auroc", "detection_rate",
                      "false_alarm_rate", "tile_auroc", "top_tile_hit_rate"]].round(3).to_string(index=False))
    print(json.dumps({"pooled": pooled, "ci": ci}, indent=1, default=float))

    # ---------------- 3. figure: per fabric, in-fabric vs zero-shot
    BLUE, GRAY, GRID = "#2a78d6", "#9aa3b2", "#e4e3df"
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    panels = (("image_auroc", "Image AUROC", (0, 1.05), 0.5),
              ("false_alarm_rate", "False alarms (good fabric flagged)", (0, 1.05), None),
              ("tile_auroc", "Tile AUROC (same image)", (0, 1.05), 0.5))
    x = np.arange(len(FABRICS)); wbar = 0.38
    for ax, (key, title, ylim, chance) in zip(axs, panels):
        a = per_fabric[per_fabric.setting == "in-fabric (Phase 4)"].set_index("fabric")[key].reindex(FABRICS)
        b = per_fabric[per_fabric.setting == "new fabric, zero-shot (LOFO)"].set_index("fabric")[key].reindex(FABRICS)
        ax.bar(x - wbar / 2, a, wbar, color=GRAY, label="In-fabric (12 good images)")
        ax.bar(x + wbar / 2, b, wbar, color=BLUE, label="New fabric, zero-shot")
        if chance is not None:
            ax.axhline(chance, color="#52514e", lw=1, ls="--")
        ndef = per_fabric[per_fabric.setting == "in-fabric (Phase 4)"].set_index("fabric").n_defect.reindex(FABRICS)
        ax.set_xticks(x, [f"{f}\n(n={n})" for f, n in zip(FABRICS, ndef)], fontsize=10)
        ax.set_ylim(*ylim); ax.set_title(title, fontsize=12)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    axs[0].set_xlabel("Fabric (n = defective test images)", fontsize=10)
    axs[0].legend(frameon=False, loc="lower left", fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIGURES / "phase5_lofo_by_fabric.png"); plt.close(fig)
    print(f"done in {time.time() - t0:.0f}s")
