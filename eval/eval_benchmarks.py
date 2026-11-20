from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from configs.defaults import EnvConfig
from envs.lf_uav_env import LowFidelityUAVEnv
from models.policy import B2LocalPowerHeuristic, B3VelocitySafetyFilter, MTCMAPPOAgent


def evaluate_policy_episodes(
    agent_or_ctrl: MTCMAPPOAgent | B2LocalPowerHeuristic | B3VelocitySafetyFilter,
    env_cfg: EnvConfig,
    num_episodes: int = 100,
    base_seed: int = 200_000,
    ground_power_override: float | None = None,
    prior_uniform_mix_override: float | None = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate a frozen checkpoint or B2/B3 comparator over `num_episodes` paired episodes."""
    env = LowFidelityUAVEnv(cfg=env_cfg, training=False)
    ep_records: List[Dict[str, float]] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        info: Dict[str, Any] = {}
        while not done:
            if isinstance(agent_or_ctrl, B2LocalPowerHeuristic):
                mot, pwr = agent_or_ctrl.act(
                    obs,
                    prev_channel_gains=env.prev_channel_gains,
                    prev_tx_powers=env.prev_tx_powers,
                    device=device,
                )
            elif isinstance(agent_or_ctrl, B3VelocitySafetyFilter):
                mot, pwr = agent_or_ctrl.act(
                    obs,
                    uav_pos=env.uav_pos,
                    uav_vel=env.uav_vel,
                    uav_caps=env.uav_caps,
                    obstacle_centers=env.obstacle_centers,
                    device=device,
                )
            else:
                act_out = agent_or_ctrl.act(
                    obs,
                    deterministic=True,
                    prior_uniform_mix_override=prior_uniform_mix_override,
                    device=device,
                )
                mot, pwr = act_out["motion_cmds"], act_out["slot_powers"]

            obs, _, _, done, info = env.step(
                mot, pwr, ground_power_override=ground_power_override
            )
        ep_records.append(info["episode_metrics"])

    def _nanmean_key(key: str) -> float:
        vals = [r[key] for r in ep_records if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else 0.0

    full_count = int(round(sum(r["full_success"] for r in ep_records)))
    cf_count = int(round(sum(r["cf_full_success"] for r in ep_records)))
    return {
        "num_episodes": num_episodes,
        "full_count": full_count,
        "cf_full_count": cf_count,
        "full_success": full_count / float(max(1, num_episodes)),
        "cf_full_success": cf_count / float(max(1, num_episodes)),
        "partial_completion": _nanmean_key("partial_completion"),
        "team_energy_kj": _nanmean_key("team_energy_kj"),
        "comm_success": _nanmean_key("comm_success"),
        "mean_sinr_db": _nanmean_key("mean_sinr_db"),
        "collision_rate": _nanmean_key("collision_rate"),
        "obs_events_per_frame": _nanmean_key("obs_events_per_frame"),
        "s_spec": _nanmean_key("s_spec"),
        "s13_alignment": _nanmean_key("s13_alignment"),
        "attempted_per_frame": _nanmean_key("attempted_per_frame"),
        "effective_per_frame": _nanmean_key("effective_per_frame"),
        "executed_steps": _nanmean_key("executed_steps"),
        "episode_records": ep_records,
    }


def evaluate_s6_allocation_diagnostics(
    agent: MTCMAPPOAgent,
    env_cfg: EnvConfig,
    num_episodes: int = 100,
    base_seed: int = 210_000,
    p_min: float = 0.02,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute Supplementary Table S6 active-sender power-allocation diagnostics (Eq. S4).

    - Uses active sender-frame records with >= 1 attempted link (P_ij > P_min).
    - Entropy H_i^t and complement C_i^t = 1 - H_i^t use active sender-frames with M_i^t >= 2,
      preserving C_ep = 1 - H_ep identically before rounding.
    """
    cfg_copy = EnvConfig(**{**env_cfg.__dict__})
    cfg_copy.wireless.p_min = p_min
    env = LowFidelityUAVEnv(cfg=cfg_copy, training=False)

    ep_entropies: List[float] = []
    ep_complements: List[float] = []
    ep_max_shares: List[float] = []
    ep_below_fracs: List[float] = []
    ep_active_powers: List[float] = []
    ep_att_per_frame: List[float] = []
    ep_eff_per_frame: List[float] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        info: Dict[str, Any] = {}
        h_list: List[float] = []
        max_sh_list: List[float] = []
        below_list: List[float] = []
        act_p_list: List[float] = []

        while not done:
            act_out = agent.act(obs, deterministic=True, device=device)
            slot_powers = act_out["slot_powers"]  # (N, M_cap)
            nbr_mask = obs["nbr_mask"]            # (N, M_cap) bool

            for i in range(env.cfg.num_uavs):
                v_slots = np.where(nbr_mask[i])[0]
                m_i = v_slots.size
                if m_i == 0:
                    continue
                p_valid = slot_powers[i, v_slots]
                if not np.any(p_valid > p_min):
                    continue  # Inactive sender-frame excluded per S6

                p_tot = float(np.sum(p_valid))
                if p_tot <= 0.0:
                    continue
                shares = p_valid / p_tot
                act_p_list.append(p_tot)
                max_sh_list.append(float(np.max(shares)))
                below_list.append(float(np.sum(p_valid <= p_min)) / float(m_i))

                if m_i >= 2:
                    pos = shares > 0.0
                    h_val = -float(np.sum(shares[pos] * np.log(shares[pos]))) / float(np.log(m_i))
                    h_val = float(np.clip(h_val, 0.0, 1.0))
                    h_list.append(h_val)

            obs, _, _, done, info = env.step(act_out["motion_cmds"], slot_powers)

        m_ep = info["episode_metrics"]
        ep_att_per_frame.append(m_ep["attempted_per_frame"])
        ep_eff_per_frame.append(m_ep["effective_per_frame"])
        if h_list:
            h_mean = float(np.mean(h_list))
            ep_entropies.append(h_mean)
            ep_complements.append(1.0 - h_mean)
        if max_sh_list:
            ep_max_shares.append(float(np.mean(max_sh_list)))
            ep_below_fracs.append(float(np.mean(below_list)))
            ep_active_powers.append(float(np.mean(act_p_list)))

    h_final = float(np.mean(ep_entropies)) if ep_entropies else 0.0
    c_final = float(np.mean(ep_complements)) if ep_complements else 0.0
    return {
        "allocation_entropy": h_final,
        "entropy_complement": c_final,
        "max_neighbor_share": float(np.mean(ep_max_shares)) if ep_max_shares else 0.0,
        "frac_below_pmin": float(np.mean(ep_below_fracs)) if ep_below_fracs else 0.0,
        "active_sender_power_w": float(np.mean(ep_active_powers)) if ep_active_powers else 0.0,
        "team_attempts_per_frame": float(np.mean(ep_att_per_frame)),
        "team_effective_per_frame": float(np.mean(ep_eff_per_frame)),
    }
