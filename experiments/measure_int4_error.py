"""Milestone 6: measure the relative INT4 quantization error of real KV tensors.

Replaces the assumed WARM quant_error=0.05 with a number measured on
SmolVLM-256M's actual prefill KV cache. For each layer, K and V are
quantized per token with asymmetric min-max INT4 in groups along head_dim
(group sizes 64 = whole head, and 32), dequantized, and compared:

    rel_err = ||x - dequant(quant(x))||_2 / ||x||_2     (per token per head)

Reported as means over tokens/heads, split by VISUAL vs TEXT tokens, plus a
K per-channel variant (KIVI-style: group along the token axis).
Writes results/int4_error.csv. Requires torch + transformers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from extract_vlm_trace import MODEL_ID, TRACES_DIR, make_images


def quant_groups(x: torch.Tensor, group: int) -> torch.Tensor:
    """x: (..., d). Asymmetric min-max INT4 per group of ``group`` along last dim."""
    shp = x.shape
    g = x.reshape(*shp[:-1], shp[-1] // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = (hi - lo).clamp_min(1e-8) / 15.0
    q = ((g - lo) / scale).round().clamp(0, 15)
    return (q * scale + lo).reshape(shp)


def rel_err(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Relative L2 error over the last dim -> (...,)."""
    return (x - xq).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-8)


def kv_layers(cache):
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager").eval()
    image_token_id = model.config.image_token_id
    rows = []
    for name, (image, question) in make_images().items():
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        inputs = processor(text=processor.apply_chat_template(msgs, add_generation_prompt=True),
                           images=[image], return_tensors="pt")
        vis = inputs["input_ids"][0] == image_token_id
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        for li, (k, v) in enumerate(kv_layers(out.past_key_values)):
            k, v = k[0], v[0]  # (kv_heads, tokens, head_dim)
            for tname, x in (("K", k), ("V", v)):
                variants = {"tok_g64": quant_groups(x, 64), "tok_g32": quant_groups(x, 32)}
                if tname == "K":  # per-channel: group along tokens
                    variants["chan_g64"] = quant_groups(x.transpose(1, 2), 64 if x.shape[1] % 64 == 0
                                                        else x.shape[1]).transpose(1, 2) \
                        if x.shape[1] % 64 == 0 else None
                for vname, xq in variants.items():
                    if xq is None:
                        continue
                    e = rel_err(x, xq).mean(0).numpy()  # mean over heads -> (tokens,)
                    rows.append(dict(image=name, layer=li, tensor=tname, scheme=vname,
                                     err_visual=float(e[vis.numpy()].mean()),
                                     err_text=float(e[~vis.numpy()].mean())))
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TRACES_DIR.parent / "results" / "int4_error.csv", index=False)
    s = df.groupby(["tensor", "scheme"])[["err_visual", "err_text"]].mean()
    print(s.round(4).to_string())
    by_layer = df[df.scheme == "tok_g32"].groupby("layer")[["err_visual", "err_text"]].mean()
    print("\nper-layer (tok_g32, K+V mean) visual err: min %.3f max %.3f" % (
        by_layer.err_visual.min(), by_layer.err_visual.max()))
    print("overall mean tok_g32 (K+V, visual): %.4f  (text): %.4f" % (
        df[df.scheme == "tok_g32"].err_visual.mean(), df[df.scheme == "tok_g32"].err_text.mean()))


if __name__ == "__main__":
    main()
