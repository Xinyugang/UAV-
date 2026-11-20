from __future__ import annotations

import numpy as np
import pytest
import torch

from configs.defaults import EnvConfig, ModelConfig
from envs.lf_uav_env import LowFidelityUAVEnv
from models.policy import B2LocalPowerHeuristic, B3VelocitySafetyFilter, MTCMAPPOAgent


@pytest.mark.parametrize(
    "method,expected_s7",
    [
        (
            "mappo_morph",
            {"actor_s7": 84600, "critic_s7": 117900, "attn_s7": 0, "shared_s7": 18300, "total_s7": 220800},
        ),
        (
            "gat_mappo",
            {"actor_s7": 93200, "critic_s7": 126800, "attn_s7": 47900, "shared_s7": 18300, "total_s7": 286200},
        ),
        (
            "morph_gat",
            {"actor_s7": 96700, "critic_s7": 132400, "attn_s7": 57300, "shared_s7": 18300, "total_s7": 304700},
        ),
        (
            "mtc_mappo",
            {"actor_s7": 97600, "critic_s7": 132700, "attn_s7": 61000, "shared_s7": 18300, "total_s7": 309600},
        ),
    ],
)
def test_table_s7_module_parameter_alignment(method: str, expected_s7: dict) -> None:
    """Verify exact module-group parameter alignment with Supplementary Table S7."""
    agent = MTCMAPPOAgent(ModelConfig(method=method))
    bd = agent.parameter_breakdown()
    for k, exp_val in expected_s7.items():
        assert bd[k] == exp_val, f"{method} {k}: got {bd[k]}, expected {exp_val}"

    # Verify strict mutual exclusivity of the 4 module groups
    actor_ids = {id(p) for p in agent.actor.parameters()}
    critic_ids = {id(p) for p in agent.critic.parameters()}
    shared_ids = {id(p) for p in agent.shared_encoders.parameters()}
    attn_ids = set() if agent.attention is None else {id(p) for p in agent.attention.parameters()}

    assert actor_ids.isdisjoint(critic_ids)
    assert actor_ids.isdisjoint(shared_ids)
    assert actor_ids.isdisjoint(attn_ids)
    assert critic_ids.isdisjoint(shared_ids)
    assert critic_ids.isdisjoint(attn_ids)
    assert shared_ids.isdisjoint(attn_ids)
    all_ids = {id(p) for p in agent.parameters()}
    assert all_ids == (actor_ids | critic_ids | shared_ids | attn_ids)


@pytest.mark.parametrize(
    "method,expected_total_s3",
    [
        ("mappo", 219400),
        ("role_mappo", 236900),
    ],
)
def test_table_s3_baseline_total_params(method: str, expected_total_s3: int) -> None:
    """Verify MAPPO and Role-MAPPO total parameter counts against Supplementary Table S3."""
    agent = MTCMAPPOAgent(ModelConfig(method=method))
    bd = agent.parameter_breakdown()
    r100_tot = int(round(bd["total_exact"] / 100.0) * 100)
    assert r100_tot == expected_total_s3


@pytest.mark.parametrize(
    "cfg_kwargs",
    [
        {"method": "mtc_mappo"},
        {"method": "morph_gat"},
        {"method": "gat_mappo"},
        {"method": "mappo_morph"},
        {"method": "role_mappo"},
        {"method": "mappo"},
        {"method": "tarmac_b1"},
        {"method": "mtc_mappo", "ablate_capability_descriptor": True},
        {"method": "mtc_mappo", "ablate_task_attention": True},
        {"method": "mtc_mappo", "ablate_link_energy_mod": True},
        {"method": "mtc_mappo", "ablate_comm_prior": True},
        {"method": "mtc_mappo", "neutral_query_override": True},
    ],
)
def test_all_baselines_and_ablations_forward(cfg_kwargs: dict) -> None:
    """Verify forward action & critic execution for all 7 methods and all Table 7 / E2 ablations."""
    env = LowFidelityUAVEnv(EnvConfig(), training=False)
    obs, critic_state = env.reset(seed=101)
    agent = MTCMAPPOAgent(ModelConfig(**cfg_kwargs))

    out = agent.act(obs, deterministic=False)
    assert out["motion_cmds"].shape == (5, 2)
    assert out["slot_powers"].shape == (5, 3)
    assert np.all(out["slot_powers"] >= 0.0)
    assert np.all(np.sum(out["slot_powers"], axis=-1) <= 1.0 + 1e-6)

    c_t = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in critic_state.items()}
    r_t = torch.as_tensor(obs["uav_roles"], dtype=torch.long)
    val = agent.critic(c_t["uav_features"], c_t["task_features"], c_t["obs_features"], uav_roles=r_t)
    assert val.ndim == 0 and torch.isfinite(val)


def test_b2_and_b3_comparators() -> None:
    """Verify Supplementary Table S32 B2 local power heuristic and B3 velocity safety filter."""
    env = LowFidelityUAVEnv(EnvConfig(), training=False)
    obs, _ = env.reset(seed=202)
    mtc = MTCMAPPOAgent(ModelConfig(method="mtc_mappo"))
    mtc.eval()

    b2 = B2LocalPowerHeuristic(mtc)
    mot2, pwr2 = b2.act(obs, env.prev_channel_gains, env.prev_tx_powers)
    assert mot2.shape == (5, 2) and pwr2.shape == (5, 3)
    assert np.all(pwr2 >= 0.0)
    assert np.all(np.sum(pwr2, axis=-1) <= 1.0 + 1e-6)
    # Invalid slots must have 0 power
    assert np.all(pwr2[~obs["nbr_mask"]] == 0.0)

    b3 = B3VelocitySafetyFilter(mtc)
    mot3, pwr3 = b3.act(obs, env.uav_pos, env.uav_vel, env.uav_caps, env.obstacle_centers)
    assert mot3.shape == (5, 2) and pwr3.shape == (5, 3)
    # B3 leaves MTC power outputs unchanged
    mtc_det = mtc.act(obs, deterministic=True)
    assert np.allclose(pwr3, mtc_det["slot_powers"])
