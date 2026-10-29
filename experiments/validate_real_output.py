"""Milestone 8b: does the simulator's quality proxy predict what the real model does?

Usage: python experiments/validate_real_output.py [n_images] [device]
For each image: (1) greedy-generate 64 tokens with SmolVLM-256M (the reference, full-precision KV);
(2) replay each policy's placement (tier schedule recorded from the simulator on the saved trace) on the
real model: at decode step t the KV entries (token, layer) that are WARM at that step are replaced by their
INT4 round trip (asymmetric min-max, groups of 32 along head_dim, K and V), everything else stays full
precision; the model is teacher-forced on the reference tokens. Reported per (image, policy): mean KL(ref || policy)
over the next-token distributions, top-1 agreement with the reference, and the simulator proxy
(0.05 x warm attention mass). Controls: random placement with the same warm count per step as the dynamic
policy, and everything-INT4. Writes results/m8_real_output.csv.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import DynamicCache

from kvtier.kv_entry import Tier
from kvtier.policy import DynamicHybridPolicy, StaticModalityPolicy
from kvtier.real_trace import load_layer_trace
from kvtier.reconfig import ReconfigurationManager
from kvtier.simulator import run_session
from kvtier.tiers import load_tier_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
N_NEW = 64
QUANT_ERROR = 0.05
PX, DX = 8.0, 0.4  # thresholds chosen by the M7 leave-one-out CV on 256M


def fake_int4(x: torch.Tensor, group: int = 32) -> torch.Tensor:
    shp = x.shape
    g = x.reshape(*shp[:-1], shp[-1] // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = (hi - lo).clamp_min(1e-8) / 15.0
    return (((g - lo) / scale).round().clamp(0, 15) * scale + lo).reshape(shp)


def schedules(trace_path: Path):
    """Warm masks per step for each policy, from the simulator, plus the simulator's quality proxy."""
    cfg = yaml.safe_load(open(CONFIG_PATH))
    tiers = load_tier_config(CONFIG_PATH)
    session = load_layer_trace(trace_path)
    n_steps = len(session.steps)
    n_layers = len(session.steps[0].new_positions)
    n_prefill = len(session.entries) - sum(len(s.new_positions) for s in session.steps)
    u = 1.0 / n_prefill
    out = {}
    for name, pol in (
        ("static", StaticModalityPolicy()),
        ("dynamic", DynamicHybridPolicy(PX * u, DX * u, warmup_steps=cfg["policy"]["dynamic_hybrid"]["warmup_steps"])),
    ):
        log: list = []
        rm = ReconfigurationManager(hot=tiers[Tier.HOT], cost_multiplier=10.0)
        r = run_session(session_copy(trace_path), pol, tiers, reconfig_manager=rm,
                        ema_decay=cfg["trace"]["ema_decay"], tier_log=log)
        out[name] = (log, r.mean_quality_loss)
    return out, n_steps, n_layers, n_prefill


def session_copy(path):  # fresh entries per simulation (they carry mutable tier state)
    return load_layer_trace(path)


def to_mask(warm_positions: set, n_tokens: int, n_layers: int) -> torch.Tensor:
    m = np.zeros((n_layers, n_tokens), dtype=bool)
    for p in warm_positions:
        tok, layer = divmod(p, n_layers)
        if tok < n_tokens:  # the step's own new token is computed inside the forward pass, in full precision
            m[layer, tok] = True
    return torch.from_numpy(m)


def proxy_from_mask(masks, weights_steps, n_layers):
    """0.05 x mean over steps of (layer-averaged attention mass landing on WARM entries)."""
    tot = 0.0
    for t, m in enumerate(masks):
        w = weights_steps[t]  # (n_layers, T_t), rows normalised
        tot += float((w[:, : m.shape[1]] * m.numpy()).sum() / n_layers)
    return QUANT_ERROR * tot / len(masks)


def main() -> None:
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    device = sys.argv[2] if len(sys.argv) > 2 else "mps"
    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv")))
    kinds = sorted({r["kind"] for r in rows})
    per_kind = {k: [r for r in rows if r["kind"] == k] for k in kinds}
    pick = []
    for i in range(max(len(v) for v in per_kind.values())):  # round-robin across kinds, stratified
        for k in kinds:
            if i < len(per_kind[k]) and len(pick) < n_images:
                pick.append(per_kind[k][i])
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
    out_path = ROOT / "results" / "m8_real_output.csv"
    out_rows = []
    rng = np.random.default_rng(0)

    for row in pick:
        t0 = time.time()
        name = row["name"]
        trace_path = ROOT / "traces" / "s256" / f"{name}_layers.npz"
        image = Image.open(ROOT / "data" / "images" / f"{name}.jpg").convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
        inputs = processor(text=processor.apply_chat_template(msgs, add_generation_prompt=True),
                           images=[image], return_tensors="pt").to(device)
        ids = inputs["input_ids"]
        L = ids.shape[1]
        sched, n_steps, n_layers, n_prefill = schedules(trace_path)
        assert n_prefill // n_layers == L - 1 and n_steps == N_NEW, (n_prefill, n_layers, L, n_steps)
        with np.load(trace_path) as z:
            lens, flat = z["lengths"], z["weights"].astype(np.float32)
        wsteps, i = [], 0
        for ln in lens:
            w = np.clip(flat[i: i + n_layers * ln].reshape(n_layers, ln), 0, None)
            wsteps.append(w / w.sum(1, keepdims=True)); i += n_layers * ln

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=N_NEW, min_new_tokens=N_NEW, do_sample=False)
            y = gen[0, L:]
            same_text = (processor.decode(y, skip_special_tokens=True).strip()
                         == (ROOT / "traces" / "s256" / f"{name}.txt").read_text().strip())
            pre = {k: (v[:, : L - 1] if k in ("input_ids", "attention_mask") else v) for k, v in inputs.items()}
            base = model(**pre, use_cache=True).past_key_values
            kv = [(l.keys.clone(), l.values.clone()) for l in base.layers]

            # warm masks per step per variant (n_layers x tokens in cache at that step = L-1+t)
            masks = {"static": [], "dynamic": [], "random_matched": [], "all_int4": []}
            prio = torch.from_numpy(rng.permutation(n_layers * (L - 1 + N_NEW)).reshape(n_layers, -1))
            for t in range(N_NEW):
                T = L - 1 + t
                masks["static"].append(to_mask(sched["static"][0][t], T, n_layers))
                md = to_mask(sched["dynamic"][0][t], T, n_layers)
                masks["dynamic"].append(md)
                k = int(md.sum())
                pr = prio[:, :T].flatten()
                thr = torch.topk(-pr.float(), k).values.min() if k else torch.tensor(0.0)
                masks["random_matched"].append((-prio[:, :T].float() >= thr) if k else torch.zeros_like(md))
                masks["all_int4"].append(torch.ones(n_layers, T, dtype=torch.bool))

            stats = {v: dict(kl=[], agree=[]) for v in masks}
            for t in range(N_NEW):
                tok = ids[:, L - 1: L] if t == 0 else y[t - 1: t].view(1, 1)
                T = L - 1 + t
                variant_logits = {}
                for v, mlist in masks.items():
                    m = mlist[t].to(device)
                    cache = DynamicCache()
                    for li, (k_, v_) in enumerate(kv):
                        sel = m[li][None, None, :, None]
                        cache.update(torch.where(sel, fake_int4(k_), k_), torch.where(sel, fake_int4(v_), v_), li)
                    variant_logits[v] = model(input_ids=tok, past_key_values=cache, use_cache=True).logits[0, -1]
                ref = model(input_ids=tok, past_key_values=base, use_cache=True)
                lr = torch.log_softmax(ref.logits[0, -1], -1)
                for v, lg in variant_logits.items():
                    lv = torch.log_softmax(lg, -1)
                    stats[v]["kl"].append(float((lr.exp() * (lr - lv)).sum()))
                    stats[v]["agree"].append(float(lr.argmax() == lv.argmax()))
                kv = [(l.keys.clone(), l.values.clone()) for l in base.layers]

        for v in masks:
            rows_v = dict(image=name, kind=row["kind"], policy=v, n_steps=N_NEW, same_text=same_text,
                          warm_frac=float(np.mean([m.float().mean() for m in masks[v]])),
                          proxy=proxy_from_mask(masks[v], wsteps, n_layers),
                          kl=float(np.mean(stats[v]["kl"])), top1_agree=float(np.mean(stats[v]["agree"])))
            out_rows.append(rows_v)
        print(f"{name}: {time.time() - t0:.0f}s same_text={same_text} " + "  ".join(
            f"{v}: kl={np.mean(stats[v]['kl']):.4f} agree={np.mean(stats[v]['agree']):.2f}" for v in masks),
            flush=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)


if __name__ == "__main__":
    main()
