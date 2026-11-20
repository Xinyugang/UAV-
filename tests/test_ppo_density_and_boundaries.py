from __future__ import annotations

import numpy as np
import torch

from configs.defaults import EnvConfig, ModelConfig
from envs.lf_uav_env import LowFidelityUAVEnv
from eval.stats import exact_two_sided_sign_flip_p, holm_step_down_adjust, paired_t_summary
from models.policy import MTCMAPPOAgent, _diag_gaussian_entropy, _diag_gaussian_log_prob, project_power_budget


def test_empty_and_single_neighbor_boundary_conditions() -> None:
    """Verify Alg. S1 Step 3 (M=0 empty set) and Manuscript p. 14 (M=1 single neighbor)."""
    env = LowFidelityUAVEnv(EnvConfig(), training=False)
    obs, _ = env.reset(seed=303)

    # Construct synthetic batch of 4 UAVs with M = 0, 1, 2, 3 valid candidates respectively
    obs["nbr_mask"] = np.array(
        [
            [False, False, False],  # UAV 0: M = 0 (empty candidate set)
            [True, False, False],   # UAV 1: M = 1 (single candidate)
            [True, True, False],    # UAV 2: M = 2
            [True, True, True],     # UAV 3: M = 3
            [False, False, False],  # UAV 4: M = 0
        ],
        dtype=bool,
    )
    obs["nbr_ids"] = np.array(
        [[-1, -1, -1], [2, -1, -1], [1, 3, -1], [0, 1, 2], [-1, -1, -1]], dtype=np.int64
    )

    agent = MTCMAPPOAgent(ModelConfig(method="mtc_mappo"))
    act_out = agent.act(obs, deterministic=False)

    # M = 0 checks (UAV 0 and UAV 4): gate=0, powers=0, prior=0, omega=0
    for empty_i in (0, 4):
        assert act_out["gate_mu"][empty_i] == 0.0
        assert np.all(act_out["slot_powers"][empty_i] == 0.0)
        assert np.all(act_out["pi_prior"][empty_i] == 0.0)
        assert np.all(act_out["omega_bar"][empty_i] == 0.0)
        assert float(act_out["l_sampled"][empty_i].item()) == 0.0
        assert torch.all(act_out["zeta_stored"][empty_i] == 0.0)

    # M = 1 check (UAV 1): allocation softmax == 1 -> candidate power == gate_mu * P_max (<= 1.0)
    expected_p1 = float(act_out["gate_mu"][1]) * agent.cfg.p_max
    assert abs(float(act_out["slot_powers"][1, 0]) - expected_p1) < 1e-6
    assert act_out["slot_powers"][1, 1] == 0.0 and act_out["slot_powers"][1, 2] == 0.0

    # Verify M=0 log_prob and entropy strictly equal motion-only Gaussian density & entropy!
    obs_t = agent.obs_dict_to_tensors(obs)
    fwd = agent._compute_features_and_prior(obs_t)
    dp = fwd["dist_params"]
    lp_eval, ent_eval = agent.evaluate_extended_actions(
        obs_t, act_out["u_sampled"], act_out["l_sampled"], act_out["zeta_stored"]
    )

    lp_mot_0 = _diag_gaussian_log_prob(
        act_out["u_sampled"][0], dp["mu_u"][0], dp["log_std_u"][0]
    ).sum()
    ent_mot_0 = _diag_gaussian_entropy(dp["log_std_u"][0]).sum()
    assert abs(float(lp_eval[0].item() - lp_mot_0.item())) < 1e-6
    assert abs(float(ent_eval[0].item() - ent_mot_0.item())) < 1e-6

    # M=1 must include gate + 1st valid neighbor residual coordinate in log_prob & entropy
    ent_mot_1 = _diag_gaussian_entropy(dp["log_std_u"][1]).sum()
    ent_gate_1 = _diag_gaussian_entropy(dp["log_std_l"][1])
    ent_b_1 = _diag_gaussian_entropy(dp["log_std_b"][1, 0])
    assert abs(float(ent_eval[1].item() - (ent_mot_1 + ent_gate_1 + ent_b_1).item())) < 1e-6


def test_power_budget_projection_properties() -> None:
    """Verify budget projection rescales ONLY when total > P_max and zeroes invalid slots."""
    cand = torch.tensor(
        [
            [0.2, 0.3, 0.9],   # Valid first 2 -> sum = 0.5 <= 1.0 -> NO rescaling!
            [0.8, 0.7, 0.5],   # All 3 valid -> sum = 2.0 > 1.0 -> rescaled by 0.5!
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])
    proj = project_power_budget(cand, mask, p_max=1.0)
    assert torch.allclose(proj[0], torch.tensor([0.2, 0.3, 0.0]), atol=1e-7)
    assert torch.allclose(proj[1], torch.tensor([0.4, 0.35, 0.25]), atol=1e-7)


def test_ppo_extended_action_ratio_unity_and_prior_gradients() -> None:
    """Verify Eqs. (35)-(37): r_i^t(theta_old) == 1.0 and gradients flow into attention & encoders."""
    env = LowFidelityUAVEnv(EnvConfig(), training=True)
    obs, _ = env.reset(seed=404)
    # Ensure at least one >=2-neighbor sender
    obs["nbr_mask"][0] = np.array([True, True, True])
    obs["nbr_ids"][0] = np.array([1, 2, 3])

    agent = MTCMAPPOAgent(ModelConfig(method="mtc_mappo"))
    act_out = agent.act(obs, deterministic=False)

    obs_t = agent.obs_dict_to_tensors(obs)
    new_lp, entropy = agent.evaluate_extended_actions(
        obs_t, act_out["u_sampled"], act_out["l_sampled"], act_out["zeta_stored"]
    )
    ratio = torch.exp(new_lp - act_out["old_log_prob"])
    assert torch.allclose(new_lp, act_out["old_log_prob"], atol=1e-6)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)
    assert torch.all(torch.isfinite(entropy))

    # Verify gradient backpropagation through b_theta = zeta_stored - a_theta^prior(o, M)
    agent.zero_grad()
    # Weight UAVs asymmetrically so sum of log_probs depends non-trivially on a_theta^prior
    adv_weights = torch.tensor([1.5, -0.7, 0.4, -1.1, 0.8], dtype=torch.float32)
    surrogate = (new_lp * adv_weights).sum()
    surrogate.backward()

    assert agent.attention is not None
    assert agent.attention.w_q.weight.grad is not None
    assert float(agent.attention.w_q.weight.grad.abs().sum().item()) > 0.0
    assert agent.attention.w_k.weight.grad is not None
    assert float(agent.attention.w_k.weight.grad.abs().sum().item()) > 0.0
    assert float(agent.shared_encoders.f_cap[0].weight.grad.abs().sum().item()) > 0.0
    assert float(agent.shared_encoders.f_demand[0].weight.grad.abs().sum().item()) > 0.0


def test_exact_sign_flip_and_holm_against_table_s28() -> None:
    """Verify exact 2^10 sign-flip test, paired t-CI, d_z, and Holm against Supplementary Table S28."""
    # Exact 10-batch 1-W counts from Supplementary Table S28 (continued; line 1997-2006)
    mtc_counts = np.array([127, 138, 127, 113, 121, 131, 140, 121, 131, 131], dtype=np.float64)
    gat_counts = np.array([122, 112, 116, 93, 113, 114, 119, 114, 120, 121], dtype=np.float64)
    diffs_pp = 100.0 * (mtc_counts - gat_counts) / 200.0

    stat = paired_t_summary(diffs_pp)
    # Table S28 text (line 2012): mean +6.80 pp, SD 3.4817, 95% CI [4.3093, 9.2907], p=0.0020, d_z=1.9531
    assert abs(stat["mean"] - 6.80) < 1e-9
    assert abs(stat["sd"] - 3.481698) < 1e-4
    assert abs(stat["ci_low"] - 4.3093) < 1e-3
    assert abs(stat["ci_high"] - 9.2907) < 1e-3
    assert abs(stat["p_sign_flip"] - (2.0 / 1024.0)) < 1e-12
    assert abs(stat["d_z"] - 1.9531) < 1e-3

    # 9-test INT family from Table S28 (lines 1968-1976): 4 tests at 2/1024 -> Holm p = 9 * 2/1024 = 0.017578 -> 0.0176
    raw_int = [2 / 1024, 2 / 1024, 2 / 1024, 2 / 1024, 10 / 1024, 32 / 1024, 138 / 1024, 124 / 1024, 256 / 1024]
    adj_int = holm_step_down_adjust(raw_int)
    assert abs(adj_int[3] - (9.0 * 2.0 / 1024.0)) < 1e-12
    assert round(adj_int[3], 4) == 0.0176
