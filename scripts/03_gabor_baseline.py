"""Phase 3 - Classical baseline: Gabor texture anomaly detection + evaluation.

Configurations compared (each fixes a failure seen in the one before; the last is MAIN):
  clahe + self      no reference: each image compared with itself
  clahe + fabric    per-fabric reference (one median/spread per filter)
  raw   + position  per-fabric, per-position reference, WITHOUT CLAHE (preprocessing ablation)
  clahe + position  per-fabric, per-position reference + per-position normal ceiling   <- MAIN

Why "position"? On defect-free CALIBRATION images, versions 1-2 put their highest
scores at the left edge of every strip, where the fixed camera sees the fabric's
selvage and the machine behind it. A position-aware reference fixes that.
No defect labels were used to design or tune any of this.

Thresholds use ONLY defect-free calibration images:
  image threshold = 95th percentile of calibration image scores (~1 false alarm in 20 good images)
  pixel threshold = 99.9th percentile of calibration pixel scores
Test set = held-out defect-free images + every defective image with a usable mask.

Run from the project folder:  python scripts/03_gabor_baseline.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from habiscan import config as C, data, gabor as G, metrics as M, preprocess as P  # noqa: E402

CONFIGS = [("clahe", "self"), ("clahe", "fabric"), ("raw", "position"), ("clahe", "position")]
MAIN = ("clahe", "position")
LABELS = {"self": "No reference", "fabric": "Per-fabric reference", "position": "Position-aware reference"}
IMG_Q, PIX_Q = 0.95, 99.9
split = pd.read_csv(C.RESULTS / "splits.csv", dtype={"fabric": str, "defect": str})
split["abs_path"] = split.path.map(lambda p: str(C.ROOT / p))
rng = np.random.default_rng(C.SEED)


def prep(img, setting):
    return P.normalize(img) if setting == "clahe" else img


def interior(w):
    m = np.zeros((C.STRIP_H, C.STRIP_W), bool)
    m[G.BORDER:-G.BORDER, G.BORDER:max(w - G.BORDER, G.BORDER + 1)] = True
    return m


def fit_reference(setting):
    """Everything 'learned' comes from FIT images (defect-free) only."""
    ref = {"fabric": {}, "position": {}}
    pooled = []
    for fab, grp in split[split.split == "fit"].groupby("fabric"):
        glob, bands, loaded = [], [], []
        for p in grp.abs_path:
            img, w = data.load_strip(p)
            e = G.energy(prep(img, setting))
            glob.append(G.sample_pixels(e, 60000, rng))
            bands.append(G.band_samples(e, 2000, rng, w))
            loaded.append((p, w))
        cat = np.concatenate(glob, axis=1)
        ref["fabric"][fab] = G.robust_stats(cat)
        pooled.append(cat[:, ::7])
        medc, madc = G.fit_positional(bands, C.STRIP_W)
        maps = [(G.anomaly_map_positional(G.energy(prep(data.load_strip(p)[0], setting)), medc, madc), w)
                for p, w in loaded]
        ref["position"][fab] = (medc, madc, G.fit_envelope(maps, C.STRIP_W))
    ref["fabric"]["pooled"] = G.robust_stats(np.concatenate(pooled, axis=1))
    return ref


def anomaly_from_energy(e, w, variant, ref, fabric):
    if variant == "self":
        a = G.anomaly_map(e, *G.self_stats(e))
    elif variant == "fabric":
        a = G.anomaly_map(e, *ref["fabric"].get(fabric, ref["fabric"]["pooled"]))
    else:
        medc, madc, env = ref["position"][fabric]
        a = G.anomaly_map_positional(e, medc, madc) / env[None, :]
    if w < C.STRIP_W:                                  # the one narrow image: ignore mirrored padding
        a[:, max(w - G.BORDER, 0):] = 0
    return a


def anomaly(img, w, setting, variant, ref, fabric):
    return anomaly_from_energy(G.energy(prep(img, setting)), w, variant, ref, fabric)


if __name__ == "__main__":
    rows, pix_rows, pix_thr, refs = [], [], {}, {}
    for setting in ("clahe", "raw"):
        variants = [v for s, v in CONFIGS if s == setting]
        t0 = time.time()
        refs[setting] = ref = fit_reference(setting)
        print(f"[{setting}] reference fitted in {time.time() - t0:.0f}s")
        calib_pix = {v: [] for v in variants}
        ordered = pd.concat([split[split.split == "calib"], split[split.split == "test"]])
        for _, r in ordered.iterrows():
            img, w = data.load_strip(r.abs_path)
            gt = data.load_mask(r["name"]) if r.is_defect else None
            inner = interior(w)
            t_e = time.time(); e = G.energy(prep(img, setting)); t_e = time.time() - t_e
            for variant in variants:
                t1 = time.time(); a = anomaly_from_energy(e, w, variant, ref, r.fabric)
                key = (setting, variant)
                rec = {"name": r["name"], "split": r.split, "is_defect": r.is_defect, "fabric": r.fabric,
                       "defect_name": r.defect_name, "setting": setting, "variant": variant,
                       "score": G.image_score(a), "seconds": t_e + time.time() - t1}
                if r.split == "calib":
                    calib_pix[variant].append(rng.choice(a[inner], 50000, replace=False))
                else:
                    if key not in pix_thr:
                        pix_thr[key] = float(np.percentile(np.concatenate(calib_pix[variant]), PIX_Q))
                    if r.is_defect:
                        pos = a[gt]; pos = rng.choice(pos, min(5000, pos.size), replace=False)
                        neg = rng.choice(a[~gt & inner], 5000, replace=False)
                        pix_rows.append((key, np.r_[pos, neg], np.r_[np.ones(pos.size), np.zeros(neg.size)]))
                        dsc, iou = M.dice_iou((a > pix_thr[key]) & inner, gt)
                        rec.update({"dice": dsc, "iou": iou, "pointing_hit": M.pointing_hit(a, gt)})
                    else:
                        neg = rng.choice(a[inner], 5000, replace=False)
                        pix_rows.append((key, neg, np.zeros(neg.size)))
                rows.append(rec)
        print(f"[{setting}] finished in {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "phase3_image_scores.csv", index=False)

    # ---------------- metrics per configuration
    table = []
    for setting, variant in CONFIGS:
        g = df[(df.setting == setting) & (df.variant == variant)]
        thr = float(np.quantile(g[g.split == "calib"].score, IMG_Q))
        test = g[g.split == "test"]
        px = [p for p in pix_rows if p[0] == (setting, variant)]
        dtest = test[test.is_defect == 1]
        table.append({"setting": setting, "variant": variant, "label": LABELS[variant],
                      **M.image_metrics(test.is_defect, test.score, thr),
                      "pixel_auroc": M.pixel_auroc(np.concatenate([p[2] for p in px]), np.concatenate([p[1] for p in px])),
                      "pixel_threshold": pix_thr[(setting, variant)],
                      "mean_dice": float(dtest.dice.mean()), "mean_iou": float(dtest.iou.mean()),
                      "pointing_hit_rate": float(dtest.pointing_hit.mean()),
                      "seconds_per_strip": float(g.seconds.mean())})
    res = pd.DataFrame(table)
    res.to_csv(C.RESULTS / "phase3_ablation.csv", index=False)
    cols = ["setting", "variant", "auroc", "f1", "precision", "recall", "false_alarm_rate", "pixel_auroc",
            "mean_dice", "mean_iou", "pointing_hit_rate", "seconds_per_strip"]
    print(res[cols].round(3).to_string(index=False))

    # ---------------- main configuration: bootstrap CIs and breakdowns
    main = df[(df.setting == MAIN[0]) & (df.variant == MAIN[1])]
    mt = main[main.split == "test"].reset_index(drop=True)
    mrow = res[(res.setting == MAIN[0]) & (res.variant == MAIN[1])].iloc[0].to_dict()
    thr = mrow["threshold"]
    boot = {"auroc": [], "f1": [], "recall": [], "precision": []}
    for _ in range(2000):
        s = mt.iloc[rng.integers(0, len(mt), len(mt))]
        if s.is_defect.nunique() < 2:
            continue
        m = M.image_metrics(s.is_defect, s.score, thr)
        for k in boot:
            boot[k].append(m[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

    d = mt[mt.is_defect == 1].assign(detected=lambda x: x.score > thr)
    by_type = d.groupby("defect_name").agg(n=("name", "size"), detected=("detected", "sum")).sort_values("n", ascending=False)
    by_type["recall"] = by_type.detected / by_type.n
    by_fabric = {}
    for fab, g in mt.groupby("fabric"):
        if g.is_defect.nunique() == 2:
            by_fabric[fab] = {"auroc": M.image_metrics(g.is_defect, g.score, thr)["auroc"],
                              "n_defect": int(g.is_defect.sum()), "n_defect_free": int((g.is_defect == 0).sum())}

    out = {"main_configuration": {"preprocessing": "CLAHE",
                                  "reference": "per-fabric, per-position defect-free statistics + per-position normal ceiling"},
           "image_threshold_rule": "95th percentile of defect-free calibration image scores",
           "pixel_threshold_rule": "99.9th percentile of defect-free calibration pixel scores",
           "test_set": {"defect_free": int((mt.is_defect == 0).sum()), "defective": int(mt.is_defect.sum())},
           "main": mrow, "bootstrap_95ci": ci,
           "recall_by_defect_type": by_type.reset_index().to_dict("records"),
           "auroc_by_fabric": by_fabric,
           "ablation": res[cols + ["threshold", "tp", "fp", "fn", "tn"]].to_dict("records")}
    (C.RESULTS / "phase3_gabor_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({"main": {k: mrow[k] for k in ("auroc", "f1", "precision", "recall", "false_alarm_rate",
                                                    "pixel_auroc", "mean_dice", "mean_iou", "pointing_hit_rate",
                                                    "seconds_per_strip", "tp", "fp", "fn", "tn")},
                      "ci95": ci, "by_fabric": by_fabric}, indent=1, default=float))
    print(by_type)

    # ---------------- figures
    INK, BLUE, ORANGE, GRAY, GRID = "#0b0b0b", "#2a78d6", "#eb6834", "#9aa3b2", "#e4e3df"
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    C.FIGURES.mkdir(parents=True, exist_ok=True)

    def clean(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color=GRID); ax.set_axisbelow(True)

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
    for variant, color in (("position", BLUE), ("fabric", ORANGE), ("self", GRAY)):
        g = df[(df.setting == "clahe") & (df.variant == variant) & (df.split == "test")]
        fpr, tpr = M.roc_points(g.is_defect, g.score)
        auc = res[(res.setting == "clahe") & (res.variant == variant)].auroc.iloc[0]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{LABELS[variant]} (AUROC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color="#c8ccd3", lw=1, ls="--")
    ax.set_xlabel("False alarm rate (good fabric flagged)"); ax.set_ylabel("Recall (defects caught)")
    ax.legend(frameon=False, loc="lower right", fontsize=11); clean(ax)
    fig.tight_layout(); fig.savefig(C.FIGURES / "phase3_roc.png"); plt.close(fig)

    cm = np.array([[mrow["tn"], mrow["fp"]], [mrow["fn"], mrow["tp"]]])
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=150)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.4)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=24, color=INK)
    ax.set_xticks([0, 1], ["Predicted PASS", "Predicted FAIL"]); ax.set_yticks([0, 1], ["Defect-free", "Defective"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(C.FIGURES / "phase3_confusion.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    bins = np.linspace(mt.score.min(), np.percentile(mt.score, 99), 40)
    ax.hist(mt[mt.is_defect == 0].score, bins=bins, color=GRAY, label="Defect-free (test)", edgecolor="white")
    ax.hist(mt[mt.is_defect == 1].score, bins=bins, color=BLUE, alpha=0.85, label="Defective (test)", edgecolor="white")
    ax.axvline(thr, color=INK, lw=1.5, ls="--"); ax.text(thr, ax.get_ylim()[1] * 0.92, "  threshold", fontsize=11)
    ax.set_xlabel("Image anomaly score"); ax.set_ylabel("Images"); ax.legend(frameon=False); clean(ax)
    fig.tight_layout(); fig.savefig(C.FIGURES / "phase3_score_distribution.png"); plt.close(fig)

    bt = by_type.sort_values("recall")
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    ax.barh([f"{n} ({int(k)})" for n, k in zip(bt.index, bt.n)], bt.recall * 100, color=BLUE, height=0.65)
    for i, v in enumerate(bt.recall * 100):
        ax.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=11)
    ax.set_xlim(0, 112); ax.set_xlabel("Defects caught at the threshold (%)  ·  (n) = test images"); clean(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(C.FIGURES / "phase3_recall_by_type.png"); plt.close(fig)

    caught = d[d.detected].sort_values("score", ascending=False).head(3).assign(kind="Caught")
    missed = d[~d.detected].sort_values("score").head(3).assign(kind="Missed")
    pick = pd.concat([caught, missed])
    names = split.set_index("name")
    fig, axs = plt.subplots(2, len(pick), figsize=(2.7 * len(pick), 5.8), dpi=150, squeeze=False)
    for j, (_, r) in enumerate(pick.iterrows()):
        img, w = data.load_strip(names.loc[r["name"], "abs_path"])
        gt = data.load_mask(r["name"])
        a = anomaly(img, w, MAIN[0], MAIN[1], refs[MAIN[0]], r.fabric)
        ys, xs = np.where(gt); cx = int(np.clip(xs.mean(), 128, w - 128)); sl = slice(cx - 128, cx + 128)
        axs[0, j].imshow(img[:, sl], cmap="gray", vmin=0, vmax=255)
        axs[1, j].imshow(img[:, sl], cmap="gray", vmin=0, vmax=255)
        axs[1, j].imshow(np.clip(a[:, sl] / pix_thr[MAIN], 0, 1.5), cmap="inferno", alpha=0.6, vmin=0, vmax=1.5)
        axs[1, j].contour(gt[:, sl], levels=[0.5], colors=["#36d6c3"], linewidths=1.2)
        axs[0, j].set_title(f"{r['kind']}: {r.defect_name}", fontsize=11)
        for a_ in axs[:, j]:
            a_.set_xticks([]); a_.set_yticks([])
    axs[0, 0].set_ylabel("Input"); axs[1, 0].set_ylabel("Anomaly map + mask")
    fig.tight_layout(); fig.savefig(C.FIGURES / "phase3_heatmaps.png"); plt.close(fig)
    print("figures saved to", C.FIGURES)
