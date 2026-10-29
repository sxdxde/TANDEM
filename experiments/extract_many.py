"""Milestone 7: per-layer decode attention traces for the whole image set and any SmolVLM size.

Usage: python experiments/extract_many.py <tag> <model_id> [device] [dtype] [limit]
Reads data/manifest.csv + data/images, writes traces/<tag>/<name>_layers.npz in the same
format as extract_vlm_layer_trace.py (head-averaged, float16) plus traces/<tag>/<name>.txt
(the generated text, for sanity-checking that the model is really using the image).
Existing outputs are skipped, so an interrupted run resumes. Requires torch + transformers.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
N_NEW = 64


def main() -> None:
    tag, model_id = sys.argv[1], sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    dtype = getattr(torch, sys.argv[4]) if len(sys.argv) > 4 else torch.float32
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
    out_dir = ROOT / "traces" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager").to(device).eval()
    image_token_id = model.config.image_token_id
    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv")))[:limit]

    for row in rows:
        name = row["name"]
        path = out_dir / f"{name}_layers.npz"
        if path.exists():
            continue
        t0 = time.time()
        image = Image.open(ROOT / "data" / "images" / f"{name}.jpg").convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
        L = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=N_NEW, min_new_tokens=N_NEW, do_sample=False,
                                 output_attentions=True, return_dict_in_generate=True)
        steps = [torch.stack([a[0].float().mean(0)[-1] for a in layers]).cpu().numpy()
                 for layers in out.attentions[:N_NEW]]
        is_visual = (inputs["input_ids"][0, : L - 1] == image_token_id).cpu().numpy()
        n_layers = steps[0].shape[0]
        assert steps[0].shape == (n_layers, L)
        tmp = path.with_name(path.stem + ".tmp.npz")  # write then rename: a kill never leaves a half file
        np.savez_compressed(
            tmp, is_visual=is_visual,
            weights=np.concatenate([s.reshape(-1) for s in steps]).astype(np.float16),
            lengths=np.array([s.shape[1] for s in steps]), n_layers=n_layers)
        tmp.rename(path)
        text = processor.batch_decode(out.sequences[:, L:], skip_special_tokens=True)[0]
        (out_dir / f"{name}.txt").write_text(text)
        print(f"{name}: L={L} visual={int(is_visual.sum())} layers={n_layers} "
              f"{time.time() - t0:.0f}s  {text[:70]!r}", flush=True)


if __name__ == "__main__":
    main()
