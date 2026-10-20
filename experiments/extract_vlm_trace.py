"""Milestone 5: extract real decode-time attention traces from a small VLM.

Runs SmolVLM-256M-Instruct on a few locally generated images (no image
downloads), greedy-decodes ``N_NEW`` tokens, and records, for each decode
step, the attention the query token pays to every earlier position,
averaged over all layers and heads. Saved via ``kvtier.real_trace`` as
traces/<name>.npz. Requires the optional deps: torch, transformers.

Alignment (see real_trace.py): the last prompt token is decode step 0's
query, taken from the prefill pass's final attention row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

from kvtier.real_trace import save_trace

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
N_NEW = 64
TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"


def make_images() -> dict[str, tuple[Image.Image, str]]:
    rng = np.random.default_rng(0)

    chart = Image.new("RGB", (1024, 768), "white")
    d = ImageDraw.Draw(chart)
    d.line([(80, 700), (80, 60), (80, 700), (960, 700)], fill="black", width=4)
    for i, h in enumerate([220, 460, 340, 580, 150]):
        x = 140 + i * 160
        d.rectangle([x, 700 - h, x + 100, 700], fill=(40 + 40 * i, 90, 200 - 30 * i))
    d.text((300, 30), "Quarterly revenue by region", fill="black")

    scene = Image.new("RGB", (1024, 768), (135, 200, 235))
    d = ImageDraw.Draw(scene)
    d.rectangle([0, 520, 1024, 768], fill=(60, 160, 70))
    d.polygon([(200, 520), (420, 220), (640, 520)], fill=(120, 80, 50))
    d.ellipse([780, 60, 920, 200], fill=(255, 220, 40))
    d.rectangle([450, 400, 560, 520], fill=(200, 40, 40))

    noise = Image.fromarray(rng.integers(0, 255, (768, 1024, 3), dtype=np.uint8))
    d = ImageDraw.Draw(noise)
    d.rectangle([380, 280, 640, 480], fill="white", outline="black", width=6)
    d.text((440, 370), "HELLO", fill="black")

    return {
        "chart": (chart, "Describe this chart in detail."),
        "scene": (scene, "What objects can you see in this image?"),
        "noise_label": (noise, "What is written in the box?"),
    }


def main() -> None:
    TRACES_DIR.mkdir(exist_ok=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    image_token_id = model.config.image_token_id

    for name, (image, question) in make_images().items():
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        L = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=N_NEW, min_new_tokens=N_NEW, do_sample=False,
                output_attentions=True, return_dict_in_generate=True,
            )

        def avg(layers, row=None):
            # layers: tuple of (1, heads, q, kv) -> mean over layers+heads -> (q, kv)
            return torch.stack([a[0].mean(0) for a in layers]).mean(0)

        step_weights = []
        for t, layers in enumerate(out.attentions):
            m = avg(layers)
            step_weights.append(m[-1].numpy())  # t=0: prefill's last row; t>0: the single row
        is_visual = (inputs["input_ids"][0, : L - 1] == image_token_id).numpy()
        step_weights = step_weights[:N_NEW]

        assert len(step_weights[0]) == L and len(is_visual) == L - 1
        path = TRACES_DIR / f"{name}.npz"
        save_trace(path, is_visual, step_weights)
        text = processor.batch_decode(out.sequences[:, L:], skip_special_tokens=True)[0]
        print(f"{name}: L={L} visual={int(is_visual.sum())} steps={len(step_weights)} -> {path}")
        print(f"  generated: {text[:120]!r}")


if __name__ == "__main__":
    sys.exit(main())
