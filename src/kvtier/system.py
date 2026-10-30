"""System-level simulation of data placement *and* computation placement (Milestone 9).

``simulator.run_session`` charges every read to the entry's tier and sums the latencies serially. That
answers "which tier should data live in", but it has no compute resources, no shared data path and no
runtime choice of *where a read is computed*. This module adds those, as a SimPy model in which memory,
compute, data movement and the runtime's decisions interact explicitly:

* **Memory tiers** (``tiers.py``): HOT (FP16 DRAM/HBM) and WARM (INT4 compute-in-memory arrays).
* **Compute resources** (finite, shared): ``host_lanes`` conventional attention lanes, ``cim_units``
  in-memory compute units that work on WARM data in place.
* **Data movement**: ``bus_channels`` host<->memory channels. A host-computed read pulls the entry's bytes
  over the bus; a CIM read moves only a small partial result. Tier moves read from one tier and write to the
  other over the same bus and need a host lane to requantise, so reconfiguration traffic competes with
  attention traffic instead of being a lump-sum penalty.
* **Runtime decisions**: (1) *where does each entry's data live* - the placement policy (``policy.py``);
  (2) *where is each read computed* - the compute scheduler (``cfg.scheduler``): WARM data is computed in
  memory ("tier"), always on the host ("host", WARM is then only compressed storage) or on whichever side is
  estimated to finish first ("adaptive", which spills to the host when the CIM units are saturated).
* **Access regimes** (``AccessRegime``): dense reads of the whole cache each step (reusable data),
  one-pass chunked reads (streaming) and random subsets (random access), so that reuse varies.

A decode step is a barrier: every read is issued as work packets of ``packet_entries`` entries at once,
the step ends when the last packet finishes, then tier moves are issued as packets and also complete before
the next step. Energy is additive and does not depend on contention; latency is the simulated makespan.

``SystemConfig.legacy(multiplier)`` reduces the model to ``run_session`` exactly (one global server, no
compute time, no result traffic, lump-sum reconfiguration); ``tests/test_system.py`` checks that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import simpy

from kvtier.kv_entry import Tier, TokenKV
from kvtier.monitor import ImportanceMonitor
from kvtier.policy import PlacementPolicy
from kvtier.tiers import TierResource, get_tier
from kvtier.trace import TraceSession


@dataclass(frozen=True)
class SystemConfig:
    # --- resources -------------------------------------------------------------------------
    bus_channels: int = 8        # parallel host<->memory channels (data movement)
    host_lanes: int = 8          # conventional attention-compute lanes
    cim_units: int = 16          # in-memory compute units attached to the WARM tier
    packet_entries: int = 64     # entries per work packet (one packet occupies one unit at a time)
    # --- per-entry compute and movement overheads -----------------------------------------------
    host_compute_ns: float = 2.0     # attention MACs for one entry on a host lane
    dequant_ns: float = 1.0          # extra host time to dequantise an INT4 entry that was pulled to the host
    dequant_pj: float = 10.0
    cim_result_bytes: float = 32.0   # partial sum returned over the bus per CIM packet
    # --- reconfiguration (tier move) --------------------------------------------------------------
    reconfig_mode: str = "explicit"  # "explicit": read src + requantise + write dst over the bus; "multiplier": m x HOT access
    reconfig_scale: float = 1.0      # scales the explicit cost (sensitivity to the bottom-up estimate)
    reconfig_multiplier: float = 10.0
    reconfig_extra_ns: float = 14.0  # requantise + table update, on a host lane
    reconfig_extra_pj: float = 60.0
    # --- runtime computation placement --------------------------------------------------------------
    scheduler: str = "tier"          # "tier" | "host" | "adaptive"
    spill_margin: float = 0.0        # adaptive: spill when est. host finish * (1 + margin) < est. CIM finish
    serial: bool = False             # one global server (used to reproduce the legacy model)

    @staticmethod
    def legacy(multiplier: float = 10.0) -> "SystemConfig":
        return SystemConfig(host_compute_ns=0.0, dequant_ns=0.0, dequant_pj=0.0, cim_result_bytes=0.0,
                            reconfig_mode="multiplier", reconfig_multiplier=multiplier, serial=True)


@dataclass(frozen=True)
class AccessRegime:
    """Which entries a decode step actually reads.

    ``dense``: every active entry (reusable data: re-read each step, as in attention over a cached prefix).
    ``stream``: the prefill is consumed once, in order, ``ceil(n_prefill_tokens / n_steps)`` tokens per step
    (a data stream: every entry is read once, no temporal reuse); the newest token is always read.
    ``random``: a uniformly random ``random_fraction`` of entries, redrawn each step (random access: no
    locality), plus the newest token. Attention weights are the real ones; weight on unread entries is
    reported as skipped mass (identical across placement policies for a given trace and regime).
    """

    kind: str = "dense"
    random_fraction: float = 0.1
    seed: int = 0


@dataclass
class SystemResult:
    policy_name: str
    energy_access_pj: float = 0.0
    energy_reconfig_pj: float = 0.0
    latency_access_ns: float = 0.0
    latency_reconfig_ns: float = 0.0
    bus_bytes_access: float = 0.0
    bus_bytes_reconfig: float = 0.0
    quality: float = 0.0              # mean over steps of sum(weight x quant_error) over the entries read
    warm_mass: float = 0.0            # mean attention mass read from WARM
    skipped_mass: float = 0.0         # mean attention mass on entries not read
    promotions: int = 0
    demotions: int = 0
    reads: int = 0
    packets: int = 0
    warm_packets: int = 0
    spilled_packets: int = 0          # WARM packets computed on the host instead of in memory
    util_bus: float = 0.0
    util_host: float = 0.0
    util_cim: float = 0.0
    wait_ns_bus: float = 0.0
    wait_ns_host: float = 0.0
    wait_ns_cim: float = 0.0

    @property
    def energy_pj(self) -> float:
        return self.energy_access_pj + self.energy_reconfig_pj

    @property
    def latency_ns(self) -> float:
        return self.latency_access_ns + self.latency_reconfig_ns

    @property
    def bus_bytes(self) -> float:
        return self.bus_bytes_access + self.bus_bytes_reconfig


def _move_cost(entry: TokenKV, cfg: SystemConfig, hot: TierResource, warm: TierResource):
    """(bus_ns, host_ns, energy_pj, bus_bytes) of one tier move of ``entry``."""
    h_lat, h_en = hot.access_cost(entry)
    if cfg.reconfig_mode == "multiplier":
        return h_lat * cfg.reconfig_multiplier, 0.0, h_en * cfg.reconfig_multiplier, hot.effective_size_bytes(entry)
    w_lat, w_en = warm.access_cost(entry)
    s = cfg.reconfig_scale
    return (s * (h_lat + w_lat), s * cfg.reconfig_extra_ns, s * (h_en + w_en + cfg.reconfig_extra_pj),
            hot.effective_size_bytes(entry) + warm.effective_size_bytes(entry))


def move_cost_in_hot_accesses(cfg: SystemConfig, tiers: dict[Tier, TierResource], entry: TokenKV) -> float:
    """One tier move expressed as a multiple of one HOT access's energy (the legacy ``m``)."""
    return _move_cost(entry, cfg, tiers[Tier.HOT], tiers[Tier.WARM])[2] / tiers[Tier.HOT].access_cost(entry)[1]


def breakeven_accesses(cfg: SystemConfig, tiers: dict[Tier, TierResource], entry: TokenKV) -> float:
    """Reads an entry must receive after a move for the move to pay in energy: R / (E_HOT - E_WARM).

    ``R`` is the cost of one tier move; ``E_HOT - E_WARM`` is the energy saved per read by being in WARM
    (computed in memory, including its share of the partial-result traffic). Never below 1.
    """
    hot, warm = tiers[Tier.HOT], tiers[Tier.WARM]
    _, _, move_pj, _ = _move_cost(entry, cfg, hot, warm)
    result_pj = cfg.cim_result_bytes / cfg.packet_entries * hot.energy_per_byte_pj
    gain = hot.access_cost(entry)[1] - (warm.access_cost(entry)[1] + result_pj)
    return max(1.0, move_pj / gain)


def run_system_session(
    session: TraceSession,
    policy: PlacementPolicy,
    tiers: dict[Tier, TierResource] | None = None,
    cfg: SystemConfig | None = None,
    regime: AccessRegime | None = None,
    ema_decay: float = 0.9,
    estimator=None,
) -> SystemResult:
    tiers = tiers if tiers is not None else {t: get_tier(t) for t in Tier}
    cfg = cfg or SystemConfig()
    regime = regime or AccessRegime()
    hot, warm = tiers[Tier.HOT], tiers[Tier.WARM]
    monitor = ImportanceMonitor(ema_decay=ema_decay)
    env = simpy.Environment()
    res_of = {
        "bus": simpy.Resource(env, cfg.bus_channels),
        "host": simpy.Resource(env, cfg.host_lanes),
        "cim": simpy.Resource(env, cfg.cim_units),
    }
    cap = {"bus": cfg.bus_channels, "host": cfg.host_lanes, "cim": cfg.cim_units}
    glock = simpy.Resource(env, 1) if cfg.serial else None
    busy = {"bus": 0.0, "host": 0.0, "cim": 0.0}
    wait = {"bus": 0.0, "host": 0.0, "cim": 0.0}
    out = SystemResult(policy_name=policy.name)

    monitor.reset(session.entries)
    for e in session.entries:
        e.reads = 0
        e.age = 0
    n_new = sum(len(s.new_positions) for s in session.steps)
    n_prefill = len(session.entries) - n_new
    n_layers = len(session.steps[0].new_positions)
    n_pre_tokens = n_prefill // n_layers
    chunk = math.ceil(n_pre_tokens / len(session.steps))
    rng = np.random.default_rng(regime.seed)
    decode_entries = session.entries[n_prefill:]
    B = cfg.packet_entries

    active: dict[int, TokenKV] = {}
    for e in session.entries[:n_prefill]:
        e.tier = policy.initial_tier(e)
        active[e.position] = e

    def use(name: str, dur: float):
        t0 = env.now
        with res_of[name].request() as rq:
            yield rq
            wait[name] += env.now - t0
            yield env.timeout(dur)
            busy[name] += dur

    def packet(stages):
        if glock is not None:
            with glock.request() as g:
                yield g
                for name, dur in stages:
                    yield from use(name, dur)
        else:
            for name, dur in stages:
                yield from use(name, dur)

    def read_set(step):
        if regime.kind == "dense":
            return None
        newest = set(step.new_positions)
        if regime.kind == "stream":
            lo, hi = step.step * chunk, min((step.step + 1) * chunk, n_pre_tokens)
            return newest | {t * n_layers + l for t in range(lo, hi) for l in range(n_layers)}
        keys = list(step.attention_weights)
        k = max(1, round(regime.random_fraction * len(keys)))
        return newest | {keys[i] for i in rng.choice(len(keys), size=k, replace=False)}

    def process():
        cursor = 0
        t_prev = 0.0
        for step in session.steps:
            for pos in step.new_positions:
                e = decode_entries[cursor]
                cursor += 1
                e.tier = policy.initial_tier(e)
                active[pos] = e
            read = read_set(step)

            n = {Tier.HOT: 0, Tier.WARM: 0}
            lat = {Tier.HOT: 0.0, Tier.WARM: 0.0}
            en = {Tier.HOT: 0.0, Tier.WARM: 0.0}
            byt = {Tier.HOT: 0.0, Tier.WARM: 0.0}
            spill_lat = spill_en = 0.0   # cost of pulling the WARM entries' bytes through the host data path
            q = wm = skipped = 0.0
            seen: dict[int, float] = {}
            for pos, w in step.attention_weights.items():
                if read is not None and pos not in read:
                    skipped += w
                    continue
                e = active[pos]
                t = e.tier
                seen[pos] = w
                a_lat, a_en = tiers[t].access_cost(e)
                n[t] += 1
                lat[t] += a_lat
                en[t] += a_en
                size = tiers[t].effective_size_bytes(e)
                byt[t] += size
                q += w * tiers[t].quant_error
                if t == Tier.WARM:
                    wm += w
                    spill_lat += hot.base_latency_ns + hot.latency_per_byte_ns * size
                    spill_en += hot.base_energy_pj + hot.energy_per_byte_pj * size

            # --- plan the step's work packets (runtime computation placement) ---------------------
            work = {"bus": 0.0, "host": 0.0, "cim": 0.0}
            procs = []
            n_pk = {t: math.ceil(n[t] / B) for t in n}
            if n[Tier.HOT]:
                k = n_pk[Tier.HOT]
                stages = [("bus", lat[Tier.HOT] / k), ("host", cfg.host_compute_ns * n[Tier.HOT] / k)]
                procs += [env.process(packet(stages)) for _ in range(k)]
                work["bus"] += lat[Tier.HOT]
                work["host"] += cfg.host_compute_ns * n[Tier.HOT]
                out.energy_access_pj += en[Tier.HOT]
                out.bus_bytes_access += byt[Tier.HOT]
            if n[Tier.WARM]:
                k = n_pk[Tier.WARM]
                per = n[Tier.WARM] / k
                cim_stage = [("cim", lat[Tier.WARM] / k), ("bus", cfg.cim_result_bytes * hot.latency_per_byte_ns)]
                host_stage = [("bus", spill_lat / k), ("host", (cfg.host_compute_ns + cfg.dequant_ns) * per)]
                cim_pj = en[Tier.WARM] / k + cfg.cim_result_bytes * hot.energy_per_byte_pj
                host_pj = spill_en / k + cfg.dequant_pj * per
                for _ in range(k):
                    if cfg.scheduler == "tier":
                        spill = False
                    elif cfg.scheduler == "host":
                        spill = True
                    else:  # adaptive: minimum estimated finish time given the work already assigned this step
                        est_cim = (work["cim"] + cim_stage[0][1]) / cfg.cim_units
                        est_host = ((work["bus"] + host_stage[0][1]) / cfg.bus_channels
                                    + (work["host"] + host_stage[1][1]) / cfg.host_lanes)
                        spill = est_host * (1.0 + cfg.spill_margin) < est_cim
                    stages = host_stage if spill else cim_stage
                    procs.append(env.process(packet(stages)))
                    for name, dur in stages:
                        work[name] += dur
                    out.warm_packets += 1
                    if spill:
                        out.spilled_packets += 1
                        out.energy_access_pj += host_pj
                        out.bus_bytes_access += byt[Tier.WARM] / k
                    else:
                        out.energy_access_pj += cim_pj
                        out.bus_bytes_access += cfg.cim_result_bytes
            out.packets += sum(n_pk.values())
            out.reads += n[Tier.HOT] + n[Tier.WARM]
            out.quality += q
            out.warm_mass += wm
            out.skipped_mass += skipped
            if procs:
                yield env.all_of(procs)
            out.latency_access_ns += env.now - t_prev
            t_prev = env.now

            # --- importance / reuse bookkeeping, then the policy's tier moves ----------------------
            if estimator is None:
                monitor.update(active, seen)
            else:
                obs, decay = estimator.observe(step.step, seen, ema_decay)
                if obs is not None:
                    monitor.update(active, obs, decay)
            for pos in seen:
                active[pos].reads += 1
            for e in active.values():
                e.age += 1
                e.steps_since_move += 1

            moves = []
            for e in active.values():
                want = policy.update(e)
                if want != e.tier:
                    moves.append((e, want))
            if moves:
                bus_ns = host_ns = pj = 0.0
                for e, want in moves:
                    b_ns, h_ns, m_pj, m_b = _move_cost(e, cfg, hot, warm)
                    bus_ns += b_ns
                    host_ns += h_ns
                    pj += m_pj
                    out.bus_bytes_reconfig += m_b
                    if e.tier == Tier.WARM and want == Tier.HOT:
                        out.promotions += 1
                    else:
                        out.demotions += 1
                    e.tier = want
                    e.tier_changes += 1
                    e.steps_since_move = 0
                out.energy_reconfig_pj += pj
                k = math.ceil(len(moves) / B)
                stages = [("bus", bus_ns / k)] + ([("host", host_ns / k)] if host_ns else [])
                yield env.all_of([env.process(packet(stages)) for _ in range(k)])
                out.packets += k
                out.latency_reconfig_ns += env.now - t_prev
                t_prev = env.now

    env.process(process())
    env.run()
    steps = len(session.steps)
    out.quality /= steps
    out.warm_mass /= steps
    out.skipped_mass /= steps
    total = env.now or 1.0
    out.util_bus = busy["bus"] / (cap["bus"] * total)
    out.util_host = busy["host"] / (cap["host"] * total)
    out.util_cim = busy["cim"] / (cap["cim"] * total)
    out.wait_ns_bus, out.wait_ns_host, out.wait_ns_cim = wait["bus"], wait["host"], wait["cim"]
    return out
