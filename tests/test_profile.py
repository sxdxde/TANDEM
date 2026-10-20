import numpy as np
import pytest

from kvtier.kv_entry import Modality, Tier
from kvtier.policy import LayerModalityPolicy
from kvtier.profile import average_profiles, group_importance, hot_groups
from kvtier.real_trace import trace_from_layer_attention


def _session():
    rng = np.random.default_rng(0)
    is_visual = np.array([False, True, True, True])
    steps = [rng.random((3, 4 + t + 1)) for t in range(3)]
    steps[0][1, :] = 0.0
    steps[0][1, 0] = 1.0  # layer 1 attends only to token 0 (text)
    return trace_from_layer_attention(is_visual, steps)


def test_entries_carry_layer_index():
    s = _session()
    assert {e.layer for e in s.entries} == {0, 1, 2}


def test_profile_covers_groups_and_policy_places_by_group():
    s = _session()
    prof = group_importance(s)
    assert set(prof) == {(m, l) for m in Modality for l in range(3)}
    hot = hot_groups(prof, theta=float(np.median(list(prof.values()))))
    pol = LayerModalityPolicy(hot)
    for e in s.entries[:12]:
        assert pol.initial_tier(e) is (Tier.HOT if (e.modality, e.layer) in hot else Tier.WARM)
        assert pol.update(e) is e.tier


def test_average_profiles_is_mean():
    a = {(Modality.TEXT, 0): 1.0}
    b = {(Modality.TEXT, 0): 3.0, (Modality.VISUAL, 0): 2.0}
    avg = average_profiles([a, b])
    assert avg[(Modality.TEXT, 0)] == pytest.approx(2.0)
    assert avg[(Modality.VISUAL, 0)] == pytest.approx(1.0)


def test_role_layer_grouping_roles():
    from kvtier.profile import n_prefill_tokens, role_layer

    s = _session()
    n_tok = n_prefill_tokens(s)
    assert n_tok == 4
    roles = {role_layer(e, n_tok)[0] for e in s.entries}
    assert roles == {"sink", "visual", "generated"}  # no mid-prompt text in this toy trace
