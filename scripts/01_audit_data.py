"""Phase 1 - Data audit: count images, masks, defect sizes; save results/audit.json.

Run from the project folder:  python scripts/01_audit_data.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from habiscan import config as C, data  # noqa: E402

items = data.list_items()
defects = [i for i in items if i.is_defect]
normals = [i for i in items if not i.is_defect]

no_mask, empty_mask, areas = [], [], []
for it in defects:
    m = data.load_mask(it.name)
    if m is None:
        no_mask.append(it.name)
    elif not m.any():
        empty_mask.append(it.name)
    else:
        areas.append(m.mean() * 100)
areas = np.array(areas)

audit = {
    "defect_free_images": len(normals),
    "defective_images": len(defects),
    "total_images": len(items),
    "defective_with_usable_mask": int(len(areas)),
    "no_mask": no_mask,
    "empty_mask": empty_mask,
    "fabrics_defect_free": dict(sorted(Counter(i.fabric for i in normals).items())),
    "fabrics_defective": dict(sorted(Counter(i.fabric for i in defects).items())),
    "defect_types": {C.DEFECT_NAMES.get(k, k): v for k, v in Counter(i.defect for i in defects).most_common()},
    "defect_area_pct": {"median": float(np.median(areas)), "max": float(areas.max()),
                        "share_under_0.1pct": float((areas < 0.1).mean())},
}
C.RESULTS.mkdir(parents=True, exist_ok=True)
(C.RESULTS / "audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
