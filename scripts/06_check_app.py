"""Phase 6 check: the app's inspector gives the same answers as the Phase 4 evaluation.

For every calibration and test image:
  - with the fabric KNOWN, the app's score must equal the Phase 4 score
  - with AUTO-DETECT, we record which fabric it picked and whether the
    PASS/FAIL verdict changes
Writes results/phase6_app_check.json. Run:  python scripts/06_check_app.py  (~4 min)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
from habiscan import config as C, data  # noqa: E402
from habiscan.inspect_fabric import Inspector  # noqa: E402

I = Inspector()
split = pd.read_csv(C.RESULTS / "splits.csv", dtype={"fabric": str})
p4 = pd.read_csv(C.RESULTS / "phase4_image_scores_test_position.csv", dtype={"fabric": str}).set_index("name")
rows = []
for _, r in split[split.split.isin(["calib", "test"])].iterrows():
    img = data.load_gray(C.ROOT / r.path)
    auto = I.inspect(img, "auto")
    row = {"name": r["name"], "split": r.split, "fabric": r.fabric, "is_defect": int(r.is_defect),
           "auto_fabric": auto.fabric, "auto_score": auto.score, "seconds": auto.seconds}
    if r.split == "test":
        row["known_score"] = I.inspect(img, r.fabric).score
        row["phase4_score"] = float(p4.loc[r["name"], "score"])
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(C.RESULTS / "phase6_app_check.csv", index=False)
t = df[df.split == "test"]
thr = I.image_threshold


def counts(col):
    pred, y = t[col] > thr, t.is_defect == 1
    return {"tp": int((pred & y).sum()), "fp": int((pred & ~y).sum()), "fn": int((~pred & y).sum()), "tn": int((~pred & ~y).sum())}


out = {"max_abs_difference_known_fabric_vs_phase4": float((t.known_score - t.phase4_score).abs().max()),
       "verdicts_known_fabric": counts("known_score"), "verdicts_auto_detect": counts("auto_score"),
       "verdicts_phase4": counts("phase4_score"),
       "auto_detect_correct_fabric": {"defect_free_images": float((df[df.is_defect == 0].auto_fabric == df[df.is_defect == 0].fabric).mean()),
                                      "defective_images": float((df[df.is_defect == 1].auto_fabric == df[df.is_defect == 1].fabric).mean())},
       "auto_detect_by_fabric": df.assign(ok=df.auto_fabric == df.fabric).groupby("fabric").ok.mean().to_dict(),
       "seconds_per_strip_with_auto_detect": float(df.seconds.mean())}
(C.RESULTS / "phase6_app_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
