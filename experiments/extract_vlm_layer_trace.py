"""Milestone 6: extract per-layer decode attention (head-averaged) from SmolVLM.

Same model / images / prompts / greedy decoding as extract_vlm_trace.py, but
keeps the layer axis: for each decode step, an array of shape
(n_layers, kv_len) holding the head-averaged attention row of the query
token. Saved as traces/<name>_layers.npz (float16). Requires torch + transformers.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from extract_vlm_trace import MODEL_ID, N_NEW, TRACES_DIR, make_images


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    image_token_id = model.config.image_token_id

    for name, (image, question) in make_images().items():
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        L = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=N_NEW, min_new_tokens=N_NEW,
                                 do_sample=False, output_attentions=True,
                                 return_dict_in_generate=True)
        steps = []
        for layers in out.attentions[:N_NEW]:
            # each layer: (1, heads, q, kv) -> head-mean, last query row -> (kv,)
            steps.append(torch.stack([a[0].mean(0)[-1] for a in layers]).numpy())
        is_visual = (inputs["input_ids"][0, : L - 1] == image_token_id).numpy()
        assert steps[0].shape == (len(layers), L)
        np.savez_compressed(
            TRACES_DIR / f"{name}_layers.npz",
            is_visual=is_visual,
            weights=np.concatenate([s.reshape(-1) for s in steps]).astype(np.float16),
            lengths=np.array([s.shape[1] for s in steps]),
            n_layers=len(layers),
        )
        print(f"{name}: L={L} layers={len(layers)} steps={len(steps)}", flush=True)


if __name__ == "__main__":
    main()
