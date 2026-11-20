from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np

from configs.defaults import EnvConfig
from envs.lf_uav_env import LowFidelityUAVEnv
from eval.stats import holm_step_down_adjust, paired_t_summary
from models.policy import MTCMAPPOAgent


def _build_tracked_slot_suppress_mask(
    obs: Dict[str, np.ndarray], tracked_uav_by_sender: Dict[int, int]
) -> np.ndarray:
    """Construct boolean slot_suppress_mask (N, M_cap) from UAV-ID-tracked masks.

    Follows Manuscript Section 5.1 (p. 16) & Supplementary S45:
      - Selected target is tracked by UAV ID, not candidate-array index.
      - If that UAV ID leaves the candidate set, no replacement is masked.
      - If that UAV ID returns to the candidate set, masking resumes automatically.
    """
    nbr_ids = obs["nbr_ids"]    # (N, M_cap)
    nbr_mask = obs["nbr_mask"]  # (N, M_cap)
    sup = np.zeros_like(nbr_mask, dtype=bool)
    for sender_i, target_uid in tracked_uav_by_sender.items():
        if target_uid < 0:
            continue
        match = nbr_mask[sender_i] & (nbr_ids[sender_i] == target_uid)
        sup[sender_i] = match
    return sup


def evaluate_lf_attention_masking(
    agent: MTCMAPPOAgent,
    env_cfg: EnvConfig,
    num_episodes: int = 100,
    base_seed: int = 410_000,
    conditions: Sequence[str] = (
        "no_mask",
        "mask_top",
        "mask_low",
        "random_mask",
        "randomize_attention",
    ),
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """Run Main Fig. 5 attention-masking evaluation with paired exogenous random inputs."""
    out: Dict[str, Dict[str, float]] = {}
    env = LowFidelityUAVEnv(cfg=env_cfg, training=False)

    for cond in conditions:
        full_list: List[float] = []
        comm_list: List[float] = []
        spec_list: List[float] = []
        rand_rng = np.random.Generator(np.random.PCG64(base_seed + 77))

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=base_seed + ep)
            tracked_targets: Dict[int, int] = {}
            done = False
            info: Dict[str, Any] = {}

            while not done:
                if cond in ("mask_top", "mask_low", "random_mask"):
                    # One-time target UAV-ID selection per sender on its first >=2-candidate frame
                    probe = agent.act(obs, deterministic=True, device=device)
                    omega = probe["omega_bar"]  # (N, M_cap)
                    nbr_mask = obs["nbr_mask"]
                    nbr_ids = obs["nbr_ids"]

                    for i in range(env.cfg.num_uavs):
                        if i in tracked_targets:
                            continue
                        valid_slots = np.where(nbr_mask[i])[0]
                        if valid_slots.size < 2:
                            continue
                        w_v = omega[i, valid_slots]
                        uids_v = nbr_ids[i, valid_slots]
                        if cond == "mask_top":
                            max_w = np.max(w_v)
                            ties = uids_v[np.isclose(w_v, max_w, atol=1e-12)]
                            tracked_targets[i] = int(np.min(ties))
                        elif cond == "mask_low":
                            min_w = np.min(w_v)
                            ties = uids_v[np.isclose(w_v, min_w, atol=1e-12)]
                            tracked_targets[i] = int(np.min(ties))
                        else:
                            tracked_targets[i] = int(rand_rng.choice(uids_v))

                    sup_mask = _build_tracked_slot_suppress_mask(obs, tracked_targets)
                    act_out = agent.act(
                        obs, deterministic=True, slot_suppress_mask=sup_mask, device=device
                    )
                elif cond == "randomize_attention":
                    act_out = agent.act(
                        obs, deterministic=True, attention_mode="randomize", device=device
                    )
                elif cond == "no_mask":
                    act_out = agent.act(obs, deterministic=True, device=device)
                else:
                    raise ValueError(f"Unknown masking condition: {cond}")

                obs, _, _, done, info = env.step(act_out["motion_cmds"], act_out["slot_powers"])

            m = info["episode_metrics"]
            full_list.append(m["full_success"])
            comm_list.append(m["comm_success"])
            spec_list.append(m["s_spec"])

        out[cond] = {
            "full_success": float(np.mean(full_list)),
            "comm_success": float(np.mean(comm_list)),
            "s_spec": float(np.mean(spec_list)),
        }
    return out


def evaluate_distance_matched_masking(
    checkpoints: Sequence[MTCMAPPOAgent],
    env_cfg: EnvConfig,
    episodes_per_ckpt: int = 200,
    distance_match_tol_m: float = 5.0,
    base_seed: int = 450_000,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run Supplementary Tables S45-S46 state-cloned Top vs. Distance-Matched masking control.

    Protocol (Supplementary p. 41, S45-S46):
      1. Advance unperturbed trajectory until first step t* where a sender i has both a
         top-attention candidate j_top (ties broken by smallest UAV ID) and another valid
         candidate j_dist with |d_{i,j_dist} - d_{i,j_top}| <= 5.0 m (ordered by |Delta d|, then ID).
      2. Save & clone complete simulator + RNG state at t* into three branches:
         'unmasked', 'top_masked', 'distance_matched'.
      3. Track masked target by UAV ID (no replacement if it leaves; resume if it returns).
      4. Summarize per-checkpoint rates and compute all 3 paired contrasts with Holm adjustment.
    """
    per_ckpt: List[Dict[str, Any]] = []
    unmasked_rates: List[float] = []
    top_rates: List[float] = []
    dist_rates: List[float] = []

    for ckpt_idx, agent in enumerate(checkpoints):
        env = LowFidelityUAVEnv(cfg=env_cfg, training=False)
        unmasked_succ = 0
        top_succ = 0
        dist_succ = 0
        eligible_eps = 0
        excluded_no_match = 0

        for ep in range(episodes_per_ckpt):
            obs, _ = env.reset(seed=base_seed + 10_000 * ckpt_idx + ep)
            done = False
            info: Dict[str, Any] = {}
            branched = False

            while not done:
                probe = agent.act(obs, deterministic=True, device=device)
                omega = probe["omega_bar"]
                nbr_mask = obs["nbr_mask"]
                nbr_ids = obs["nbr_ids"]
                nbr_d_m = obs["nbr_dist_cost"] * env.cfg.d_norm  # metres

                # Locate first eligible (sender_i, top_uid, dist_matched_uid)
                match_tuple = None
                for i in range(env.cfg.num_uavs):
                    v_slots = np.where(nbr_mask[i])[0]
                    if v_slots.size < 2:
                        continue
                    w_v = omega[i, v_slots]
                    uids_v = nbr_ids[i, v_slots]
                    dists_v = nbr_d_m[i, v_slots]

                    max_w = np.max(w_v)
                    top_candidates = np.where(np.isclose(w_v, max_w, atol=1e-12))[0]
                    # Break ties by smallest UAV ID
                    top_local_idx = top_candidates[int(np.argmin(uids_v[top_candidates]))]
                    top_uid = int(uids_v[top_local_idx])
                    top_d = float(dists_v[top_local_idx])

                    other_choices = []
                    for loc_k in range(v_slots.size):
                        cand_uid = int(uids_v[loc_k])
                        if cand_uid == top_uid:
                            continue
                        diff_d = abs(float(dists_v[loc_k]) - top_d)
                        if diff_d <= distance_match_tol_m:
                            other_choices.append((diff_d, cand_uid))
                    if other_choices:
                        other_choices.sort(key=lambda item: (item[0], item[1]))
                        match_tuple = (i, top_uid, other_choices[0][1])
                        break

                if match_tuple is not None:
                    branched = True
                    eligible_eps += 1
                    sender_i, top_uid, dist_uid = match_tuple
                    snap_t_star = env.clone_state()

                    def _run_branch(tracked_map: Dict[int, int]) -> int:
                        b_obs = env.restore_state(snap_t_star)[0]
                        b_done = False
                        b_info: Dict[str, Any] = {}
                        while not b_done:
                            if tracked_map:
                                sup = _build_tracked_slot_suppress_mask(b_obs, tracked_map)
                                b_act = agent.act(
                                    b_obs, deterministic=True, slot_suppress_mask=sup, device=device
                                )
                            else:
                                b_act = agent.act(b_obs, deterministic=True, device=device)
                            b_obs, _, _, b_done, b_info = env.step(
                                b_act["motion_cmds"], b_act["slot_powers"]
                            )
                        return int(round(b_info["episode_metrics"]["full_success"]))

                    unmasked_succ += _run_branch({})
                    top_succ += _run_branch({sender_i: top_uid})
                    dist_succ += _run_branch({sender_i: dist_uid})
                    break

                obs, _, _, done, info = env.step(probe["motion_cmds"], probe["slot_powers"])

            if not branched:
                excluded_no_match += 1

        denom = float(max(1, eligible_eps))
        r_un = unmasked_succ / denom
        r_top = top_succ / denom
        r_dm = dist_succ / denom
        unmasked_rates.append(r_un)
        top_rates.append(r_top)
        dist_rates.append(r_dm)
        per_ckpt.append(
            {
                "checkpoint_index": ckpt_idx,
                "eligible_episodes": eligible_eps,
                "excluded_no_match": excluded_no_match,
                "unmasked_count": unmasked_succ,
                "top_masked_count": top_succ,
                "distance_matched_count": dist_succ,
                "unmasked_rate": r_un,
                "top_masked_rate": r_top,
                "distance_matched_rate": r_dm,
            }
        )

    # Contrasts in percentage points (pp), matching Supplementary Table S46
    diff_top_un = [100.0 * (t - u) for t, u in zip(top_rates, unmasked_rates)]
    diff_dm_un = [100.0 * (d - u) for d, u in zip(dist_rates, unmasked_rates)]
    diff_top_dm = [100.0 * (t - d) for t, d in zip(top_rates, dist_rates)]

    s1 = paired_t_summary(diff_top_un)
    s2 = paired_t_summary(diff_dm_un)
    s3 = paired_t_summary(diff_top_dm)
    p_holm = holm_step_down_adjust([s1["p_sign_flip"], s2["p_sign_flip"], s3["p_sign_flip"]])
    s1["p_holm"] = p_holm[0]
    s2["p_holm"] = p_holm[1]
    s3["p_holm"] = p_holm[2]

    return {
        "mean_unmasked": float(np.mean(unmasked_rates)) if unmasked_rates else 0.0,
        "mean_top_masked": float(np.mean(top_rates)) if top_rates else 0.0,
        "mean_distance_matched": float(np.mean(dist_rates)) if dist_rates else 0.0,
        "per_checkpoint": per_ckpt,
        "contrasts_pp": {
            "top_minus_unmasked": s1,
            "distance_matched_minus_unmasked": s2,
            "top_minus_distance_matched": s3,
        },
    }
