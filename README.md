# HabiScan — automated fabric defect inspection

CSST 106 (Perception and Computer Vision) final project. HabiScan looks at an image
of woven fabric, decides whether it is defective, and shows where the defect is.
Data: the AITEX Fabric Image Database (Silvestre-Blanes et al., 2019).

## Project status

| Phase | What | Status |
|---|---|---|
| 1 | Data audit | Done (`scripts/01_audit_data.py`) |
| 2 | Preprocessing + leakage-safe split | Done (`scripts/02_preprocess.py`) |
| 3 | Classical baseline (Gabor texture) + evaluation | Done (`scripts/03_gabor_baseline.py`, `scripts/03b_shortcut_and_tile_eval.py`) |
| 4 | Deep model (PatchCore) + same evaluation | Done (`scripts/04_patchcore.py`) |
| 5 | Leave-one-fabric-out evaluation | Done (`scripts/05_leave_one_fabric_out.py`) |
| 6 | App & demo (Gradio) | Done (`app.py`, check: `scripts/06_check_app.py`) |

## Quick start: just the demo app (no dataset needed)

The trained memory banks (`models/`) and three sample images (`samples/`) are in the repo,
so the app runs right after cloning:

```
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860 and click a sample. Re-running the evaluation (scripts 01-07)
needs the full dataset: see "Getting the data".

## Getting the data

The AITEX images (~180 MB) are not stored in this repo. Download the AITEX Fabric Image
Database (Silvestre-Blanes et al., 2019). Our copy came from the GitHub repository
`jraad/fabric_defect_detection` (MIT licence). Put the three folders here:

```
data/aitex/Defect_images/      106 defective strips (e.g. 0001_002_00.png)
data/aitex/Mask_images/        expert masks (e.g. 0001_002_00_mask.png)
data/aitex/NODefect_images/    7 sub-folders of defect-free strips, one per fabric
```

Then run the scripts below in order. `results/splits.csv` fixes which image is fit /
calibration / test, so your numbers match ours.

## Setup (Windows, VS Code)

Open this folder in VS Code, then in the terminal:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. No GPU needed. The first run of Phase 4 downloads the
ImageNet ResNet-18 weights (~45 MB) into `models/` from the timm GitHub release.

## Run

From this folder, in order:

```
python scripts/01_audit_data.py               # counts, masks, defect sizes   (~10 s)
python scripts/02_preprocess.py               # split + tiling check + figure (~30 s)
python scripts/03_gabor_baseline.py           # Gabor baseline, 4 versions    (~8 min)
python scripts/03b_shortcut_and_tile_eval.py  # shortcut check + tile test    (~6 min)
python scripts/04_patchcore.py                # PatchCore + comparison        (~7 min)
python scripts/05_leave_one_fabric_out.py     # new-fabric (zero-shot) test   (~5 min)
python scripts/06_check_app.py                # app == Phase 4 check          (~4 min)
python scripts/07_slide_figures.py            # confusion matrix + error chart (~5 s)
```

Everything is written to `results/` (numbers) and `results/figures/` (charts).

## Folder layout

```
habiscan/            the code
  config.py          paths and settings (split sizes, CLAHE, tile size)
  data.py            loading images/masks, the fit/calib/test split
  preprocess.py      CLAHE contrast normalization, tiling
  gabor.py           Gabor texture features and anomaly maps
  metrics.py         AUROC, F1, Dice/IoU, pointing hit
  patchcore.py       PatchCore: ResNet-18 features, coreset memory bank, position ceiling
  inspect_fabric.py  one-image inspection used by the app
app.py               the Gradio demo app
scripts/             one script per phase (run in order)
samples/             3 test images for the app demo
data/aitex/          the dataset (not in the repo, see "Getting the data")
models/              PatchCore memory banks + position ceilings (ResNet-18 weights download on first run)
results/             metrics (JSON/CSV) and figures
```

## How the evaluation works

- **Split by image, never by tile.** Defect-free images per fabric: 8 *fit*
  (to learn what normal fabric looks like), 4 *calib* (only to choose the
  threshold), the rest *test*. Every defective image with a usable mask is test.
  (`results/splits.csv`)
- **Thresholds come only from defect-free calibration images.** Image threshold
  = 95th percentile of their scores (about 1 false alarm per 20 good images).
  No defect labels are used to design or tune anything.
- **Shortcut check.** In AITEX, several fabrics' defective images were captured
  brighter, with more contrast, or sharper than their defect-free images, so an
  image-level detector can separate the two groups without seeing the defect.
  `03b` erases the defect and re-scores, and runs a **tile-level test** inside
  each defective image (defect tiles vs. clean tiles of the same image).
  Same image = same lighting, so only the defect can make a tile stand out.
  This tile-level test is the main metric from Phase 4 on.

## Phase 3 results (test set: 104 defective, 57 defect-free images)

| Gabor version | Image AUROC* | F1* | Tile AUROC | Top tile = defect | Pixel IoU | CPU s/strip |
|---|---|---|---|---|---|---|
| No reference | 0.60 | 0.34 | 0.62 | 2% (chance 11%) | 0.000 | 0.46 |
| Per-fabric reference | 0.59 | 0.00 | 0.53 | 2% (chance 11%) | 0.000 | 0.40 |
| Position-aware, no CLAHE | 0.85 | 0.73 | 0.62 | 13% (chance 11%) | 0.015 | 0.43 |
| **Position-aware + CLAHE (main)** | **0.85** | **0.78** | **0.61** | **18% (chance 11%)** | **0.007** | **0.41** |

\* Inflated by the shortcut: with the defect erased, the main version's image AUROC
only drops from 0.85 to 0.83, and all 68 flagged defect images stay flagged.

**Conclusion:** the classical baseline cannot reliably find these tiny defects.
That is the gap Phase 4 (PatchCore) must close, judged on the tile-level test.

### Why "position-aware"?
The first two versions put their highest scores at the left edge of every strip
(checked on defect-free calibration images only): the fixed camera always sees
the fabric's edge and the machine behind it there. Comparing each column with
defect-free images *at the same position* removed that. A minimum spread and a
per-position "normal ceiling" fixed two further artifacts seen on the same
calibration images.

## Phase 4 results: PatchCore (same test set, same rules)

PatchCore: a frozen ImageNet ResNet-18 (never retrained) describes every 8x8-px patch
of every 256x256 tile (layer2 + layer3 features). Each fabric gets a memory bank of
normal patches from its 8 defect-free *fit* images (1% greedy coreset, ~1,300 patches).
A patch's score is its distance to the nearest normal patch.

Plain PatchCore had the same left-edge artifact as the first Gabor versions (on
defect-free calibration images its hottest tile was tile 0, 1 or 4 in 22 of 28).
Fix, using defect-free fit images only: score half of them against a bank built from the
other half, record how high a normal score gets at each position, and divide by it.

| Method | Image AUROC* | F1* | False alarms | Tile AUROC | Top tile = defect | Pixel AUROC | Pixel IoU | CPU s/strip |
|---|---|---|---|---|---|---|---|---|
| Gabor (Phase 3 main) | 0.85 | 0.78 | 3/57 | 0.61 | 18% | 0.74 | 0.007 | 0.41 |
| PatchCore, plain | 0.91 | 0.79 | 2/57 | 0.58 | 23% | 0.65 | 0.010 | 0.38 |
| **PatchCore + position ceiling (main)** | **0.94** | **0.81** | **3/57** | **0.68** | **30%** | **0.76** | **0.015** | **0.38** |

Chance for "top tile = defect" is 11%. 95% bootstrap CIs for the main PatchCore:
image AUROC 0.90-0.97, F1 0.75-0.87, tile AUROC 0.61-0.75, top tile 21-38%.

\* Still inflated by the capture shortcut: with the defect erased, PatchCore's image
AUROC only drops from 0.94 to 0.90, and 72 of the 73 flagged defect images stay flagged.

**Conclusion:** PatchCore beats the Gabor baseline on every shortcut-proof measure
(tile AUROC 0.68 vs 0.61; hottest tile on the defect 30% vs 18% of images), and it
finds long, high-contrast defects well (broken pick 89%, broken yarn 50%). It still
misses most tiny, low-contrast ones (nep 0%, broken end 0%, fuzzyball 23%), so the
image-level PASS/FAIL mostly reflects how the defective images were captured, not the
defect itself. Files: `results/phase4_*.json|csv`, `results/figures/phase4_*.png`.

## Phase 5 results: a fabric HabiScan has never seen

Zero-shot leave-one-fabric-out: for each fabric, the memory bank, position ceiling and
threshold come only from the other 6 fabrics. Same test images as Phase 4.

| Setting | Detection rate | False alarms | Accuracy | Image AUROC (mean per fabric†) | Tile AUROC | Top tile = defect |
|---|---|---|---|---|---|---|
| In-fabric (12 good images of that fabric) | 70% (73/104) | 5% (3/57) | 79% | 0.92 | 0.68 | 30% |
| New fabric, zero-shot | 93% (97/104) | **84% (48/57)** | 66% | **0.63** | 0.75 | 34% |

† fabrics with >= 5 defective test images (all except 05). 95% bootstrap CIs, zero-shot:
false alarms 75-93%, accuracy 59-73%. Zero-shot minus in-fabric: tile AUROC -0.01 to +0.15,
top-tile hit -6 to +14 points, so localization **holds up**, it does not clearly improve.

**Conclusion:** without any images of the new fabric, HabiScan flags almost everything as
defective and its PASS/FAIL ranking drops to near chance on some fabrics (04: 0.49, 06: 0.40).
Where it looks inside a strip still works. So a new fabric needs a short setup step: about
12 defect-free images (8 for the memory bank, 4 for the threshold), no defect labels.
Files: `results/phase5_*`, `results/figures/phase5_lofo_by_fabric.png`.

## Phase 6: the demo app

```
python app.py
```
then open **http://localhost:7860**. (Needs Phase 4's `models/` folder; first start
takes ~10 s to load the model.) Close it with Ctrl+C in the terminal.

What it shows: PASS/FAIL, anomaly score (1.0 = as high as normal fabric gets),
defect area %, which fabric memory was used, and a heatmap with the hottest tile
boxed. The threshold slider re-judges the same image instantly (no re-scoring).

Demo samples (test images, never used for training or thresholds):

| Sample | File | Score | Verdict at 1.47 |
|---|---|---|---|
| Clean fabric | `samples/clean_0018_000_03.png` (fabric 03) | 0.93 | PASS |
| Broken yarn | `samples/broken_yarn_0081_006_04.png` | 2.14 | FAIL, box on the yarn |
| Tiny fuzzyball (hard case) | `samples/fuzzyball_0045_019_04.png` | 1.13 | PASS (missed). Slide to 1.10 -> FAIL, box on the fuzzyball |

**Checked** (`scripts/06_check_app.py`): with the fabric known, the app reproduces the
Phase 4 scores (max difference 0.0002) and the same 73 / 3 / 31 / 54 verdicts on the
161 test images. **Auto-detect** picks the right fabric for 100% of defect-free images
but only 74% of defective ones (fabric 02 defect images often look closest to 04);
the PASS/FAIL verdicts were still identical. The app shows which fabric it used and
lets you override it.

Limits: built for AITEX-style grayscale line-scan strips (256 px tall). Phone photos or
resized crops look unlike anything in memory and will usually FAIL.

## Data notes (from the audit)
- 247 images in this copy (the paper says 245): 141 defect-free, 106 defective.
- 104 defective images have a usable mask: `0100_025_08` has no mask and
  `0106_010_03` has an empty one (both excluded). Two masks are split into
  `_mask1`/`_mask2` files and are merged.
- `0094_027_05` is 3796 px wide instead of 4096; it is padded by mirroring and
  the padding is ignored when scoring.
- Median defect covers about 0.01% of the image; 77% cover less than 0.1%.

## Credits
- Dataset: J. Silvestre-Blanes, T. Albero-Albero, I. Miralles, R. Pérez-Llorens,
  J. Moreno (2019). *A public fabric database for defect detection methods and
  results.* AUTEX Research Journal 19(4). doi:10.2478/aut-2019-0035.
  This copy was taken from GitHub `jraad/fabric_defect_detection` (MIT licence,
  see `data/aitex/SOURCE_REPO_LICENSE.txt`).
