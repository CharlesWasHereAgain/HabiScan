"""HabiScan - Phase 6 demo app.

Upload a fabric image (or pick a sample) -> PASS / FAIL, anomaly score,
heatmap with the hottest tile boxed, defect area %, and a threshold slider.
Model = Phase 4 PatchCore + position ceiling (nothing is retrained here).

Run from the project folder (after Phase 4 has produced models/):
    pip install -r requirements.txt
    python app.py
then open http://localhost:7860 in a browser.
"""
from pathlib import Path

import gradio as gr
import numpy as np

from habiscan.inspect_fabric import FABRICS, Inspector

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "samples"   # copies of 3 AITEX test images, so the demo works without the full dataset
SAMPLES = [  # test-split images (never used to build or tune the model)
    ("Clean fabric (expected PASS)", SAMPLE_DIR / "clean_0018_000_03.png"),
    ("Broken yarn (expected FAIL)", SAMPLE_DIR / "broken_yarn_0081_006_04.png"),
    ("Tiny fuzzyball (the hard case)", SAMPLE_DIR / "fuzzyball_0045_019_04.png"),
]
FABRIC_CHOICES = ["Auto-detect"] + [f"Fabric {f}" for f in FABRICS]

print("Loading model (first run downloads ~45 MB of ResNet-18 weights)...")
INSPECTOR = Inspector()
DEFAULT_THR = round(INSPECTOR.image_threshold, 2)


def report(r, thr):
    v = INSPECTOR.verdict(r, thr)
    verdict = "## ❌ FAIL — defect suspected" if v["fail"] else "## ✅ PASS — looks like normal fabric"
    how = "auto-detected" if r.auto else "chosen by you"
    details = (f"**Anomaly score:** {v['score']:.2f} (threshold {thr:.2f}; 1.0 ≈ as high as normal fabric gets)  \n"
               f"**Defect area:** {v['defect_area_pct']:.3f}% of the fabric  \n"
               f"**Hottest tile:** {v['hottest_tile'] + 1} of 16 (boxed when FAIL)  \n"
               f"**Fabric memory used:** {r.fabric} ({how})  \n"
               f"**Time:** {r.seconds:.1f} s on CPU")
    if r.auto:
        ranked = sorted(r.fit_distance.items(), key=lambda kv: kv[1])
        details += "  \n<sub>Closest fabrics: " + ", ".join(f"{f} ({d:.2f})" for f, d in ranked[:3]) + \
                   ". If this is wrong, pick the fabric yourself: a wrong fabric raises false alarms.</sub>"
    return verdict, details, INSPECTOR.overlay(r, v)


def run(image, fabric_choice, thr):
    if image is None:
        return None, "Upload an image or pick a sample.", "", None
    fabric = "auto" if fabric_choice == "Auto-detect" else fabric_choice.split()[-1]
    r = INSPECTOR.inspect(np.asarray(image), fabric)
    verdict, details, pic = report(r, thr)
    return r, verdict, details, pic


def rethreshold(r, thr):
    if r is None:
        return gr.update(), gr.update(), gr.update()
    return report(r, thr)


def load_sample(name):
    path = dict(SAMPLES)[name]
    from habiscan.data import load_gray
    return load_gray(path)


with gr.Blocks(title="HabiScan") as demo:
    gr.Markdown("# HabiScan — fabric defect inspection\n"
                "Trained only on defect-free fabric (AITEX dataset). Upload a grayscale fabric strip "
                "(best: 256 px tall, like AITEX) or pick a sample.")
    state = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(label="Fabric image", type="numpy", image_mode="L", format="png", height=160)  # png: lossless (webp would change pixels)
            sample = gr.Radio([s[0] for s in SAMPLES], label="…or pick a sample")
            fabric = gr.Dropdown(FABRIC_CHOICES, value="Auto-detect", label="Fabric type")
            thr = gr.Slider(1.0, 3.0, value=DEFAULT_THR, step=0.01,
                            label=f"PASS/FAIL threshold (calibrated: {DEFAULT_THR})")
            go = gr.Button("Inspect", variant="primary")
        with gr.Column(scale=2):
            verdict = gr.Markdown()
            details = gr.Markdown()
    heatmap = gr.Image(label="Heatmap (bright = unlike normal fabric; box = hottest tile)", format="png", height=220)
    gr.Markdown("<sub>Model: frozen ImageNet ResNet-18 + PatchCore memory of normal patches, one per fabric. "
                "Threshold set on defect-free calibration images only. On the 161 test images: 73/104 defects "
                "caught, 3/57 good images flagged.</sub>")

    sample.change(load_sample, sample, image).then(run, [image, fabric, thr], [state, verdict, details, heatmap])
    go.click(run, [image, fabric, thr], [state, verdict, details, heatmap])
    thr.change(rethreshold, [state, thr], [verdict, details, heatmap])  # fast: no re-scoring

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
