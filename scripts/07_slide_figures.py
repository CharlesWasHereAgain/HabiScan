"""Slide figures for the Evaluation slide (reads Phase 4 results only; ~5 s).

  results/figures/phase4_confusion_labeled.png  PASS/FAIL confusion matrix + accuracy/precision/recall/F1
  results/figures/phase4_errors_by_type.png     error analysis: image flagged vs. hottest tile on the defect
Run:  python scripts/07_slide_figures.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from habiscan import config as C  # noqa: E402

d = json.loads((C.RESULTS / "phase4_patchcore_metrics.json").read_text())
m = d["main"]
BLUE, GRAY, GRID, INK = "#2a78d6", "#9aa3b2", "#e4e3df", "#0b0b0b"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})

# 1. labeled confusion matrix
tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
acc = (tp + tn) / (tp + fp + fn + tn)
cm = np.array([[tn, fp], [fn, tp]])
labels = [["good fabric passed", "false alarm"], ["defect missed", "defect caught"]]
fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.4)
for i in range(2):
    for j in range(2):
        ax.text(j, i - 0.08, str(int(cm[i, j])), ha="center", va="center", fontsize=34, color=INK)
        ax.text(j, i + 0.22, labels[i][j], ha="center", va="center", fontsize=13, color="#333333")
ax.set_xticks([0, 1], ["Predicted PASS", "Predicted FAIL"], fontsize=14)
ax.set_yticks([0, 1], ["Defect-free\n(57)", "Defective\n(104)"], fontsize=14)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(f"Accuracy {acc:.0%}   Precision {m['precision']:.0%}   Recall {m['recall']:.0%}   F1 {m['f1']:.2f}",
             fontsize=14, pad=14)
fig.tight_layout(); fig.savefig(C.FIGURES / "phase4_confusion_labeled.png"); plt.close(fig)

# 2. error analysis by defect type (types with >= 5 test images)
rec = {r["defect_name"]: (r["n"], r["recall"]) for r in d["recall_by_defect_type"]}
top = {r["defect_name"]: r["top_tile_hit_rate"] for r in d["top_tile_hit_by_defect_type"]}
names = [k for k, (n, _) in sorted(rec.items(), key=lambda kv: -kv[1][0]) if n >= 5]
a = [rec[n][1] * 100 for n in names]; b = [top[n] * 100 for n in names]
fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
x = np.arange(len(names)); w = 0.38
ax.bar(x - w / 2, a, w, color=GRAY, label="Image flagged FAIL")
ax.bar(x + w / 2, b, w, color=BLUE, label="Hottest tile is on the defect")
for i, (u, v) in enumerate(zip(a, b)):
    ax.text(i - w / 2, u + 1.5, f"{u:.0f}%", ha="center", fontsize=11)
    ax.text(i + w / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=11)
ax.set_xticks(x, [f"{n}\n(n={rec[n][0]})" for n in names], fontsize=11)
ax.set_ylim(0, 108); ax.set_ylabel("% of test images")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(C.FIGURES / "phase4_errors_by_type.png"); plt.close(fig)
print("saved", C.FIGURES / "phase4_confusion_labeled.png", C.FIGURES / "phase4_errors_by_type.png")
