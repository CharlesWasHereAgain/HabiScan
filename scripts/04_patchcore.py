"""Phase 4 - Deep model: PatchCore, evaluated exactly like the Gabor baseline.

Same split (results/splits.csv), same CLAHE input, same threshold rules, same
metrics, plus the two shortcut-proof checks from Phase 3b:
  - image threshold = 95th percentile of defect-free CALIBRATION image scores
  - pixel threshold = 99.9th percentile of defect-free CALIBRATION pixel scores
  - erase test: remove the defect (32-px dilation) and re-score the image
  - tile-level test: defect tiles vs. clean tiles INSIDE each defective image
    (same image = same lighting, so only the defect can make a tile stand out)
No defect labels are used to build or tune anything.

Two versions are compared (the second fixes a failure seen in the first):
  plain     PatchCore as published: nearest-neighbour distance to the fabric's bank
  position  plain score / per-position normal ceiling (fitted on defect-free FIT
            images only), because on defect-free CALIBRATION images the plain
            version's hottest tile sits at the strip's left edge   <- MAIN

Run from the project folder (after Phases 1-3):
    python scripts/04_patchcore.py          (~10 min on a laptop CPU)
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from habiscan import config as C, data, gabor as G, metrics as M, patchcore as PC  # noqa: E402

IMG_Q, PIX_Q, TILE_TOPK = 0.95, 99.9, 50
ERASE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))     # 32-px dilation, as in 03b
rng = np.random.default_rng(C.SEED)
torch.manual_seed(C.SEED)
split = pd.read_csv(C.RESULTS / "splits.csv", dtype={"fabric": str, "defect": str})
split["abs_path"] = split.path.map(lambda p: str(C.ROOT / p))


def interior(w):
    m = np.zeros((C.STRIP_H, C.STRIP_W), bool)
    m[PC.BORDER:-PC.BORDER, PC.BORDER:max(w - PC.BORDER, PC.BORDER + 1)] = True
    return m


def tile_scores(a, w):
    """Score of each 256x256 tile = mean of its top-50 pixels (identical to 03b)."""
    out = []
    for i in range(C.STRIP_W // C.TILE):
        x0, x1 = i * C.TILE + PC.BORDER, min((i + 1) * C.TILE, w) - PC.BORDER
        if x1 - x0 < 32:
            out.append(np.nan); continue
        t = a[PC.BORDER:-PC.BORDER, x0:x1].ravel()
        out.append(float(np.partition(t, -TILE_TOPK)[-TILE_TOPK:].mean()))
    return np.array(out)


if __name__ == "__main__":
    t0 = time.time()
    model = PC.load_backbone()

    # ---------------- 1. memory banks: one per fabric, defect-free FIT images only
    banks = {}
    for fab, grp in split[split.split == "fit"].groupby("fabric"):
        t1 = time.time()
        banks[fab] = PC.fit_bank(model, [data.load_strip(p) for p in grp.abs_path])
        print(f"fabric {fab}: memory bank {tuple(banks[fab].shape)} in {time.time() - t1:.0f}s")
    envs = {}
    for fab, grp in split[split.split == "fit"].groupby("fabric"):
        t1 = time.time()
        envs[fab] = PC.fit_envelope(model, [data.load_strip(p) for p in grp.abs_path])
        print(f"fabric {fab}: position ceiling in {time.time() - t1:.0f}s")
    (C.ROOT / "models").mkdir(exist_ok=True)
    PC.save_banks(banks, C.ROOT / "models" / "patchcore_banks.pt")
    np.savez(C.ROOT / "models" / "patchcore_position_ceiling.npz", **envs)
    VARIANTS = {"plain": "PatchCore (plain)", "position": "PatchCore + position ceiling"}
    MAIN = "position"

    def variant_map(pm, fab, variant):
        return pm if variant == "plain" else pm / envs[fab][None, :]

    # ---------------- 2. score calibration + test images
    rows, maps, calib_pix = [], {}, {v: [] for v in VARIANTS}
    ordered = pd.concat([split[split.split == "calib"], split[split.split == "test"]])
    for _, r in ordered.iterrows():
        img, w = data.load_strip(r.abs_path)
        t1 = time.time()
        pm = PC.patch_map(PC.features(model, img), banks[r.fabric], w)
        sec = time.time() - t1
        maps[r["name"]] = pm
        for v in VARIANTS:
            a = PC.pixel_map(variant_map(pm, r.fabric, v), w)
            rows.append({"name": r["name"], "split": r.split, "is_defect": r.is_defect, "fabric": r.fabric,
                         "defect_name": r.defect_name, "variant": v, "score": G.image_score(a), "seconds": sec})
            if r.split == "calib":
                calib_pix[v].append(rng.choice(a[interior(w)], 50000, replace=False))
    print(f"scored {len(rows)} images in {time.time() - t0:.0f}s total")
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "phase4_image_scores.csv", index=False)
    np.savez_compressed(C.RESULTS / "phase4_patch_maps.npz", **{k: v.astype(np.float16) for k, v in maps.items()})

    def evaluate(variant):
        dv = df[df.variant == variant]
        img_thr = float(np.quantile(dv[dv.split == "calib"].score, IMG_Q))
        pix_thr = float(np.percentile(np.concatenate(calib_pix[variant]), PIX_Q))
        test = dv[dv.split == "test"].reset_index(drop=True)

        # ---------------- 3. pixel metrics, erase test, tile-level test (defective test images)
        names = split.set_index("name")
        pix_s, pix_l, extra, erased, t_lab, t_sc, tile_rows = [], [], {}, {}, [], [], []
        for _, r in test.iterrows():
            row = names.loc[r["name"]]
            _, w = data.load_strip(row.abs_path)
            a = PC.pixel_map(variant_map(maps[r["name"]], r.fabric, variant), w)
            inner = interior(w)
            if not r.is_defect:
                neg = rng.choice(a[inner], 5000, replace=False)
                pix_s.append(neg); pix_l.append(np.zeros(neg.size)); continue
            gt = data.load_mask(r["name"])
            pos = a[gt]; pos = rng.choice(pos, min(5000, pos.size), replace=False)
            neg = rng.choice(a[~gt & inner], 5000, replace=False)
            pix_s.append(np.r_[pos, neg]); pix_l.append(np.r_[np.ones(pos.size), np.zeros(neg.size)])
            dsc, iou = M.dice_iou((a > pix_thr) & inner, gt)
            extra[r["name"]] = {"dice": dsc, "iou": iou, "pointing_hit": M.pointing_hit(a, gt)}
            a_er = a.copy(); a_er[cv2.dilate(gt.astype(np.uint8), ERASE) > 0] = 0
            erased[r["name"]] = G.image_score(a_er)
            ts = tile_scores(a, w)
            lab = np.array([gt[:, i * C.TILE:(i + 1) * C.TILE].any() for i in range(C.STRIP_W // C.TILE)])
            ok = ~np.isnan(ts); ts, lab = ts[ok], lab[ok]
            tile_rows.append({"name": r["name"], "defect_name": r.defect_name, "fabric": r.fabric,
                              "top_tile_has_defect": bool(lab[np.argmax(ts)]), "chance": float(lab.mean()),
                              "defect_tiles": int(lab.sum()), "tiles": int(lab.size)})
            t_lab.extend(lab.astype(int)); t_sc.extend(ts)
        for k in ("dice", "iou", "pointing_hit"):
            test[k] = test.name.map(lambda n: extra.get(n, {}).get(k, np.nan))
        test.to_csv(C.RESULTS / f"phase4_image_scores_test_{variant}.csv", index=False)
        tr = pd.DataFrame(tile_rows)
        tr.to_csv(C.RESULTS / f"phase4_tile_hits_{variant}.csv", index=False)

        dtest = test[test.is_defect == 1]
        main = {"label": VARIANTS[variant], **M.image_metrics(test.is_defect, test.score, img_thr),
                "pixel_auroc": M.pixel_auroc(np.concatenate(pix_l), np.concatenate(pix_s)),
                "pixel_threshold": pix_thr, "mean_dice": float(dtest.dice.mean()), "mean_iou": float(dtest.iou.mean()),
                "pointing_hit_rate": float(dtest.pointing_hit.mean()),
                "seconds_per_strip": float(dv.seconds.mean())}
        normal_test = test[test.is_defect == 0].score.to_numpy()
        er = np.array([erased[n] for n in dtest.name])
        shortcut = {
            "image_auroc_original": main["auroc"],
            "image_auroc_with_defects_erased": float(M.image_metrics(
                np.r_[np.zeros(normal_test.size), np.ones(er.size)], np.r_[normal_test, er], img_thr)["auroc"]),
            "defect_images_flagged_original": int((dtest.score > img_thr).sum()),
            "defect_images_still_flagged_after_erase": int((er > img_thr).sum()),
            "tile_auroc_within_defect_images": M.pixel_auroc(np.array(t_lab), np.array(t_sc)),
            "top_tile_hit_rate": float(tr.top_tile_has_defect.mean()),
            "top_tile_chance_rate": float(tr.chance.mean()),
            "n_defect_images": int(len(tr)),
        }

        # ---------------- 4. bootstrap CIs and breakdowns (same as Phase 3)
        boot = {"auroc": [], "f1": [], "recall": [], "precision": []}
        tile_boot, top_boot = [], []
        tl, tsc = np.array(t_lab), np.array(t_sc)
        img_idx = np.repeat(np.arange(len(tr)), tr.tiles.to_numpy())
        for _ in range(2000):
            s = test.iloc[rng.integers(0, len(test), len(test))]
            if s.is_defect.nunique() == 2:
                m = M.image_metrics(s.is_defect, s.score, img_thr)
                for k in boot:
                    boot[k].append(m[k])
            pick = rng.integers(0, len(tr), len(tr))                       # resample defective IMAGES
            sel = np.concatenate([np.where(img_idx == i)[0] for i in pick])
            tile_boot.append(M.pixel_auroc(tl[sel], tsc[sel]))
            top_boot.append(tr.top_tile_has_defect.to_numpy()[pick].mean())
        ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
        ci["tile_auroc"] = [float(np.percentile(tile_boot, 2.5)), float(np.percentile(tile_boot, 97.5))]
        ci["top_tile_hit_rate"] = [float(np.percentile(top_boot, 2.5)), float(np.percentile(top_boot, 97.5))]

        d = dtest.assign(detected=lambda x: x.score > img_thr)
        by_type = d.groupby("defect_name").agg(n=("name", "size"), detected=("detected", "sum")).sort_values("n", ascending=False)
        by_type["recall"] = by_type.detected / by_type.n
        top_by_type = tr.groupby("defect_name").top_tile_has_defect.agg(["size", "mean"]).rename(
            columns={"size": "n", "mean": "top_tile_hit_rate"})
        by_fabric = {}
        for fab, g in test.groupby("fabric"):
            if g.is_defect.nunique() == 2:
                by_fabric[fab] = {"auroc": M.image_metrics(g.is_defect, g.score, img_thr)["auroc"],
                                  "n_defect": int(g.is_defect.sum()), "n_defect_free": int((g.is_defect == 0).sum())}

        return dict(main=main, shortcut=shortcut, ci=ci, by_type=by_type, top_by_type=top_by_type, by_fabric=by_fabric,
                    test=test, tr=tr, img_thr=img_thr, pix_thr=pix_thr)

    R = {v: evaluate(v) for v in VARIANTS}
    main, shortcut, ci, by_type, top_by_type, by_fabric, test, tr, img_thr, pix_thr = (
        R[MAIN][k] for k in ("main", "shortcut", "ci", "by_type", "top_by_type", "by_fabric", "test", "tr", "img_thr", "pix_thr"))

    # ---------------- 5. side-by-side with the Gabor baseline (from Phase 3 result files)
    g3 = json.loads((C.RESULTS / "phase3_gabor_metrics.json").read_text())["main"]
    g3b = json.loads((C.RESULTS / "phase3b_shortcut_and_tile_eval.json").read_text())["clahe+position"]
    keys = ["auroc", "f1", "precision", "recall", "false_alarm_rate", "pixel_auroc", "mean_dice", "mean_iou",
            "pointing_hit_rate", "seconds_per_strip"]
    sk = ["image_auroc_with_defects_erased", "tile_auroc_within_defect_images", "top_tile_hit_rate", "top_tile_chance_rate"]
    comparison = pd.DataFrame([{"method": "Gabor (position-aware + CLAHE)", **{k: g3[k] for k in keys}, **{k: g3b[k] for k in sk}},
                               *[{"method": VARIANTS[v], **{k: R[v]["main"][k] for k in keys}, **{k: R[v]["shortcut"][k] for k in sk}}
                                 for v in VARIANTS]])
    comparison.to_csv(C.RESULTS / "phase4_comparison.csv", index=False)

    out = {"method": "PatchCore + per-position normal ceiling: frozen ImageNet ResNet-18 (layer2+layer3, 3x3 avg), per-fabric memory bank, "
                     f"{PC.CORESET_RATIO:.0%} greedy coreset, nearest-neighbour distance, Gaussian sigma {PC.SMOOTH_SIGMA}",
           "image_threshold_rule": "95th percentile of defect-free calibration image scores",
           "pixel_threshold_rule": "99.9th percentile of defect-free calibration pixel scores",
           "test_set": {"defect_free": int((test.is_defect == 0).sum()), "defective": int(test.is_defect.sum())},
           "memory_bank_size_per_fabric": {k: int(v.shape[0]) for k, v in banks.items()},
           "main_variant": MAIN, "main": main, "shortcut_and_tile_eval": shortcut, "bootstrap_95ci": ci,
           "ablation": {v: {**R[v]["main"], **R[v]["shortcut"]} for v in VARIANTS},
           "recall_by_defect_type": by_type.reset_index().to_dict("records"),
           "top_tile_hit_by_defect_type": top_by_type.reset_index().to_dict("records"),
           "auroc_by_fabric": by_fabric, "runtime_seconds_total": time.time() - t0}
    (C.RESULTS / "phase4_patchcore_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(comparison.T.to_string())
    print(json.dumps({"ci95": ci, "by_fabric": by_fabric}, indent=1, default=float))
    print(by_type); print(top_by_type)

    # ---------------- 6. figures
    names = split.set_index("name")
    INK, BLUE, ORANGE, GRAY, GRID = "#0b0b0b", "#2a78d6", "#eb6834", "#9aa3b2", "#e4e3df"
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})

    def clean(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color=GRID); ax.set_axisbelow(True)

    # 6a. headline comparison: tile-level test (main metric), Gabor vs PatchCore variants
    labels = ["Gabor", "PatchCore\nplain", "PatchCore\n+ position"]
    tile_auc = [g3b["tile_auroc_within_defect_images"], R["plain"]["shortcut"]["tile_auroc_within_defect_images"],
                R["position"]["shortcut"]["tile_auroc_within_defect_images"]]
    hit = [g3b["top_tile_hit_rate"] * 100, R["plain"]["shortcut"]["top_tile_hit_rate"] * 100,
           R["position"]["shortcut"]["top_tile_hit_rate"] * 100]
    chance = g3b["top_tile_chance_rate"] * 100
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    x = np.arange(3)
    axs[0].bar(x, tile_auc, 0.55, color=[GRAY, GRAY, BLUE])
    axs[0].axhline(0.5, color="#52514e", lw=1, ls="--"); axs[0].text(2.4, 0.51, "chance", fontsize=10, ha="right")
    for i, v in enumerate(tile_auc):
        axs[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    axs[0].set_xticks(x, labels); axs[0].set_ylim(0, 1); axs[0].set_ylabel("Tile AUROC")
    axs[0].set_title("Tile-level test (same image, same lighting)", fontsize=12)
    axs[1].bar(x, hit, 0.55, color=[GRAY, GRAY, BLUE])
    axs[1].axhline(chance, color="#52514e", lw=1, ls="--"); axs[1].text(2.4, chance + 1.5, f"chance {chance:.0f}%", fontsize=10, ha="right")
    for i, v in enumerate(hit):
        axs[1].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=11)
    axs[1].set_xticks(x, labels); axs[1].set_ylim(0, 100); axs[1].set_ylabel("% of defect images")
    axs[1].set_title("Hottest tile contains the defect", fontsize=12)
    for ax in axs:
        clean(ax); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(C.FIGURES / "phase4_vs_gabor_tile_test.png"); plt.close(fig)

    # 6b. ROC, both methods (image level)
    g3s = pd.read_csv(C.RESULTS / "phase3_image_scores.csv", dtype={"fabric": str})
    g3s = g3s[(g3s.setting == "clahe") & (g3s.variant == "position") & (g3s.split == "test")]
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
    for (lab, ys, ss, col) in (("PatchCore + position", test.is_defect, test.score, BLUE),
                               ("PatchCore plain", R["plain"]["test"].is_defect, R["plain"]["test"].score, GRAY),
                               ("Gabor", g3s.is_defect, g3s.score, ORANGE)):
        fpr, tpr = M.roc_points(ys, ss)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{lab} (AUROC {M.image_metrics(ys, ss, 0)['auroc']:.2f})")
    ax.plot([0, 1], [0, 1], color="#c8ccd3", lw=1, ls="--")
    ax.set_xlabel("False alarm rate (good fabric flagged)"); ax.set_ylabel("Recall (defects caught)")
    ax.legend(frameon=False, loc="lower right", fontsize=11); clean(ax)
    fig.tight_layout(); fig.savefig(C.FIGURES / "phase4_roc_vs_gabor.png"); plt.close(fig)

    # 6c. confusion matrix
    cm = np.array([[main["tn"], main["fp"]], [main["fn"], main["tp"]]])
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=150)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.4)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=24, color=INK)
    ax.set_xticks([0, 1], ["Predicted PASS", "Predicted FAIL"]); ax.set_yticks([0, 1], ["Defect-free", "Defective"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(C.FIGURES / "phase4_confusion.png"); plt.close(fig)

    # 6d. heatmap crops: 3 where the top tile is the defect, 3 where it isn't
    hit_r = tr[tr.top_tile_has_defect].merge(test[["name", "score"]]).sort_values("score", ascending=False).head(3).assign(kind="Found")
    miss_r = tr[~tr.top_tile_has_defect].sample(min(3, (~tr.top_tile_has_defect).sum()), random_state=C.SEED).assign(kind="Missed")
    pick = pd.concat([hit_r, miss_r])
    fig, axs = plt.subplots(2, len(pick), figsize=(2.7 * len(pick), 5.8), dpi=150, squeeze=False)
    for j, (_, r) in enumerate(pick.iterrows()):
        img, w = data.load_strip(names.loc[r["name"], "abs_path"])
        gt = data.load_mask(r["name"]); a = PC.pixel_map(variant_map(maps[r["name"]], r.fabric, MAIN), w)
        ys_, xs_ = np.where(gt); cx = int(np.clip(xs_.mean(), 128, w - 128)); sl = slice(cx - 128, cx + 128)
        axs[0, j].imshow(img[:, sl], cmap="gray", vmin=0, vmax=255)
        axs[1, j].imshow(img[:, sl], cmap="gray", vmin=0, vmax=255)
        axs[1, j].imshow(np.clip(a[:, sl] / pix_thr, 0, 1.5), cmap="inferno", alpha=0.6, vmin=0, vmax=1.5)
        axs[1, j].contour(gt[:, sl], levels=[0.5], colors=["#36d6c3"], linewidths=1.2)
        axs[0, j].set_title(f"{r['kind']}: {r.defect_name}", fontsize=11)
        for a_ in axs[:, j]:
            a_.set_xticks([]); a_.set_yticks([])
    axs[0, 0].set_ylabel("Input"); axs[1, 0].set_ylabel("PatchCore map + mask")
    fig.tight_layout(); fig.savefig(C.FIGURES / "phase4_heatmaps.png"); plt.close(fig)

    # 6e. full strips, hottest tile boxed (same layout as phase3b_localization_examples)
    fig, axs = plt.subplots(len(pick), 1, figsize=(14, 1.1 * len(pick)), dpi=150)
    for ax, (_, r) in zip(np.atleast_1d(axs), pick.iterrows()):
        img, w = data.load_strip(names.loc[r["name"], "abs_path"])
        gt = data.load_mask(r["name"]); a = PC.pixel_map(variant_map(maps[r["name"]], r.fabric, MAIN), w)
        best = int(np.nanargmax(tile_scores(a, w)))
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.imshow(np.clip(a / pix_thr, 0, 1.5), cmap="inferno", alpha=0.45, vmin=0, vmax=1.5, aspect="auto")
        ax.contour(gt, levels=[0.5], colors=["#36d6c3"], linewidths=1.5)
        ax.add_patch(plt.Rectangle((best * C.TILE, 0), C.TILE - 1, C.STRIP_H - 1, fill=False, ec="#ffffff", lw=2))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{r['kind']}\n{r.defect_name}", rotation=0, ha="right", va="center", fontsize=10)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.995, bottom=0.005, hspace=0.08)
    fig.savefig(C.FIGURES / "phase4_localization_examples.png"); plt.close(fig)

    # 6f. top-tile hit by defect type
    tb = top_by_type.sort_values("top_tile_hit_rate")
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    ax.barh([f"{n} ({int(k)})" for n, k in zip(tb.index, tb.n)], tb.top_tile_hit_rate * 100, color=BLUE, height=0.65)
    for i, v in enumerate(tb.top_tile_hit_rate * 100):
        ax.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=11)
    ax.set_xlim(0, 112); ax.set_xlabel("Hottest tile contains the defect (%)  ·  (n) = test images"); clean(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(C.FIGURES / "phase4_top_tile_by_type.png"); plt.close(fig)
    print(f"done in {time.time() - t0:.0f}s; figures in", C.FIGURES)
