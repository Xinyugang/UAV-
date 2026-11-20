from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from scipy.stats import rankdata

from configs.defaults import CAP_NORM_REF, EnvConfig, S29_DEMAND_CONDITIONS
from envs.lf_uav_env import LowFidelityUAVEnv
from envs.service_and_reward import compute_c_pure_reference, compute_routing_weights
from eval.stats import holm_step_down_adjust, paired_t_summary
from models.policy import MTCMAPPOAgent


# Seven-point discrete Spearman support for M in {2, 3} with average-rank ties (Table S50)
SEVEN_POINT_SPEARMAN_SUPPORT: Tuple[Tuple[str, float], ...] = (
    ("-1", -1.0),
    ("-sqrt3/2", -math.sqrt(3.0) / 2.0),
    ("-0.5", -0.5),
    ("0", 0.0),
    ("0.5", 0.5),
    ("sqrt3/2", math.sqrt(3.0) / 2.0),
    ("1", 1.0),
)


def compute_within_state_spearman(
    x: Sequence[float], y: Sequence[float], const_tol: float = 1e-12
) -> Tuple[float | None, str | None]:
    """Compute within-state Spearman correlation using average ranks for ties (Table S44 & S50).

    Exclusion rules (Supplementary p. 39, S44):
      - Fewer than 2 candidates -> 'fewer_than_2'
      - Non-finite entries -> 'nonfinite'
      - Constant vector (max - min <= 1e-12) -> 'constant_vector'
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2 or ya.size < 2 or xa.size != ya.size:
        return None, "fewer_than_2"
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        return None, "nonfinite"
    if float(np.max(xa) - np.min(xa)) <= const_tol or float(np.max(ya) - np.min(ya)) <= const_tol:
        return None, "constant_vector"

    rx = rankdata(xa, method="average")
    ry = rankdata(ya, method="average")
    dx = rx - np.mean(rx)
    dy = ry - np.mean(ry)
    denom = float(np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2)))
    if denom <= 1e-15:
        return 0.0, None
    rho = float(np.clip(np.sum(dx * dy) / denom, -1.0, 1.0))
    return rho, None


def bucket_seven_point_spearman(rho_values: Sequence[float]) -> Dict[str, float]:
    """Classify valid within-state M in {2, 3} Spearman values onto Table S50's 7-point support."""
    labels = [lbl for lbl, _ in SEVEN_POINT_SPEARMAN_SUPPORT]
    support = np.array([val for _, val in SEVEN_POINT_SPEARMAN_SUPPORT], dtype=np.float64)
    counts = np.zeros(len(labels), dtype=np.int64)
    arr = np.asarray(rho_values, dtype=np.float64)
    if arr.size == 0:
        return {lbl: 0.0 for lbl in labels}

    for r in arr:
        idx = int(np.argmin(np.abs(support - float(r))))
        counts[idx] += 1
    pcts = 100.0 * counts.astype(np.float64) / float(arr.size)
    return {lbl: float(p) for lbl, p in zip(labels, pcts)}


def evaluate_s44_candidate_correlations(
    checkpoints: Sequence[MTCMAPPOAgent],
    env_cfg: EnvConfig,
    states_per_model: int = 600,
    behavior_states_per_model: int = 200,
    behavior_horizon: int = 50,
    gamma: float = 0.99,
    base_seed: int = 540_000,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run Supplementary Tables S44, S49, and S50 candidate-level attention & behavior diagnostics.

    1. Collects `states_per_model` valid states (2 or 3 candidates) per checkpoint, saving
       state snapshots for the first `behavior_states_per_model` states.
    2. Evaluates within-state Spearman correlation between head-averaged attention bar_omega_ij
       and C_ij^ref across all 7 demand conditions D0--D6 (Table S29) with geometry, capabilities,
       candidate order, link history, and routing weights held fixed!
    3. Records population SD (divided by M) of C_ij^ref (Table S49) and 7-point discrete
       Spearman distributions (Table S50).
    4. Evaluates 50-step discounted return loss Delta G_j = G_base - G_mask(j) (with zero
       bootstrap) via exact simulator state cloning for the 'Behavior' test row of Table S44.
    5. Applies exact 2^n sign-flip tests and complete 8-test Holm adjustment (Table S44).
    """
    demand_keys = ("D0", "D1", "D2", "D3", "D4", "D5", "D6")
    cap_ref = np.asarray(CAP_NORM_REF, dtype=np.float64)

    model_demand_rhos: Dict[str, List[float]] = {k: [] for k in demand_keys}
    model_ref_sds: Dict[str, List[float]] = {k: [] for k in demand_keys}
    model_seven_pt: Dict[str, List[Dict[str, float]]] = {k: [] for k in demand_keys}
    excluded_counts: Dict[str, int] = {k: 0 for k in demand_keys}
    valid_counts: Dict[str, int] = {k: 0 for k in demand_keys}
    model_behavior_rhos: List[float] = []

    for m_idx, agent in enumerate(checkpoints):
        env = LowFidelityUAVEnv(cfg=env_cfg, training=False)
        collected_records: List[Dict[str, Any]] = []
        ep_cursor = 0

        while len(collected_records) < states_per_model:
            obs, _ = env.reset(seed=base_seed + 10_000 * m_idx + ep_cursor)
            ep_cursor += 1
            done = False
            while not done and len(collected_records) < states_per_model:
                alpha_q, _, u_unf = compute_routing_weights(
                    env.uav_pos,
                    env.task_centers,
                    env.task_progress,
                    env.task_req,
                    gamma_d=env.cfg.gamma_d,
                    gamma_u=env.cfg.gamma_u,
                    d_norm=env.cfg.d_norm,
                )
                for i in range(env.cfg.num_uavs):
                    v_slots = np.where(obs["nbr_mask"][i])[0]
                    if v_slots.size >= 2 and len(collected_records) < states_per_model:
                        uids = obs["nbr_ids"][i, v_slots]
                        cand_caps = obs["nbr_z"][i, v_slots] * cap_ref[None, :]
                        cand_e_j = obs["nbr_x"][i, v_slots, 4] * env.cfg.e_max_ref_j
                        # Require non-identical candidate capabilities/energies so C^ref is nonconstant
                        if float(np.std(cand_caps[:, 2]) + np.std(cand_caps[:, 4]) + np.std(cand_e_j)) > 1e-6:
                            need_snap = len(collected_records) < behavior_states_per_model
                            rec = {
                                "sender_i": i,
                                "obs_single": {k: v[i : i + 1].copy() for k, v in obs.items()},
                                "alpha_i": alpha_q[i].copy(),
                                "u_unf": u_unf.copy(),
                                "rel_tasks": (
                                    (env.task_centers - env.uav_pos[i][None, :]) / env.cfg.d_norm
                                ).copy(),
                                "v_slots": v_slots.copy(),
                                "uids": uids.copy(),
                                "cand_caps": cand_caps,
                                "cand_e_j": cand_e_j,
                                "env_snap": env.clone_state() if need_snap else None,
                            }
                            collected_records.append(rec)

                act_step = agent.act(obs, deterministic=True, device=device)
                obs, _, _, done, _ = env.step(act_step["motion_cmds"], act_step["slot_powers"])

        # Evaluate D0--D6 on the fixed collected states
        for d_key in demand_keys:
            d_mat = np.asarray(S29_DEMAND_CONDITIONS[d_key], dtype=np.float64)  # (3, 5)
            rhos_m: List[float] = []
            sds_m: List[float] = []

            for rec in collected_records:
                obs_mod = {k: v.copy() for k, v in rec["obs_single"].items()}
                alpha_i = rec["alpha_i"]
                # Update demand_mix and task_obs under condition d_key while holding geometry & alpha fixed
                obs_mod["demand_mix"][0] = (alpha_i @ d_mat).astype(np.float32)
                t_feats = []
                for k in range(env.cfg.num_tasks):
                    rel = rec["rel_tasks"][k]
                    d_ik = float(np.linalg.norm(rel))
                    dk = d_mat[k]
                    t_feats.extend(
                        [
                            rel[0],
                            rel[1],
                            d_ik,
                            rec["u_unf"][k],
                            alpha_i[k],
                            dk[1],
                            dk[2],
                            0.5 * (dk[3] + dk[4]),
                        ]
                    )
                obs_mod["task_obs"][0] = np.asarray(t_feats, dtype=np.float32)

                act_d = agent.act(obs_mod, deterministic=True, device=device)
                v_slots = rec["v_slots"]
                omega_v = act_d["omega_bar"][0, v_slots]
                c_ref = compute_c_pure_reference(
                    alpha_i=alpha_i,
                    task_demands=d_mat,
                    cand_caps=rec["cand_caps"],
                    cand_energy_j=rec["cand_e_j"],
                )
                # Table S49: candidate-level population SD (dividing by M)
                sds_m.append(float(np.std(c_ref, ddof=0)))

                rho_val, excl_reason = compute_within_state_spearman(omega_v, c_ref)
                if excl_reason is None and rho_val is not None:
                    rhos_m.append(rho_val)
                    valid_counts[d_key] += 1
                else:
                    excluded_counts[d_key] += 1

            model_demand_rhos[d_key].append(float(np.mean(rhos_m)) if rhos_m else 0.0)
            model_ref_sds[d_key].append(float(np.mean(sds_m)) if sds_m else 0.0)
            model_seven_pt[d_key].append(bucket_seven_point_spearman(rhos_m))

        # Evaluate 50-step discounted return loss Delta G_j = G_base - G_mask(j) on H0000..H0199
        beh_rhos_m: List[float] = []
        for rec in collected_records[:behavior_states_per_model]:
            if rec["env_snap"] is None:
                continue
            sender_i = int(rec["sender_i"])
            v_slots = rec["v_slots"]
            uids = rec["uids"]

            act_nom = agent.act(rec["obs_single"], deterministic=True, device=device)
            omega_nom = act_nom["omega_bar"][0, v_slots]

            def _rollout_discounted_return(masked_uid: int | None) -> float:
                b_obs = env.restore_state(rec["env_snap"])[0]
                g_ret = 0.0
                disc = 1.0
                for _h in range(behavior_horizon):
                    if masked_uid is not None:
                        sup = np.zeros_like(b_obs["nbr_mask"], dtype=bool)
                        sup[sender_i] = b_obs["nbr_mask"][sender_i] & (
                            b_obs["nbr_ids"][sender_i] == masked_uid
                        )
                        b_act = agent.act(
                            b_obs, deterministic=True, slot_suppress_mask=sup, device=device
                        )
                    else:
                        b_act = agent.act(b_obs, deterministic=True, device=device)
                    b_obs, _, r_step, b_done, _ = env.step(
                        b_act["motion_cmds"], b_act["slot_powers"]
                    )
                    g_ret += disc * float(r_step)
                    disc *= gamma
                    if b_done:
                        break
                return g_ret

            g_base = _rollout_discounted_return(None)
            delta_g = [g_base - _rollout_discounted_return(int(uid)) for uid in uids]
            rho_b, excl_b = compute_within_state_spearman(omega_nom, delta_g)
            if excl_b is None and rho_b is not None:
                beh_rhos_m.append(rho_b)

        model_behavior_rhos.append(float(np.mean(beh_rhos_m)) if beh_rhos_m else 0.0)

    # Eight-test Holm family (7 demand conditions + 1 Behavior row, Table S44)
    all_names = list(demand_keys) + ["Behavior"]
    summaries: List[Dict[str, float]] = []
    for name in all_names:
        vals = model_behavior_rhos if name == "Behavior" else model_demand_rhos[name]
        summaries.append(paired_t_summary(vals))

    holm_ps = holm_step_down_adjust([s["p_sign_flip"] for s in summaries])
    table_s44: Dict[str, Any] = {}
    for name, s_dict, hp in zip(all_names, summaries, holm_ps):
        s_dict["p_holm"] = hp
        table_s44[name] = s_dict

    table_s49: Dict[str, Any] = {}
    table_s50: Dict[str, Any] = {}
    ddof = 1 if len(checkpoints) > 1 else 0
    for d_key in demand_keys:
        sd_vals = model_ref_sds[d_key]
        table_s49[d_key] = {
            "models": len(checkpoints),
            "valid_states": valid_counts[d_key],
            "excluded_states": excluded_counts[d_key],
            "ref_dispersion_mean": float(np.mean(sd_vals)) if sd_vals else 0.0,
            "ref_dispersion_sd": float(np.std(sd_vals, ddof=ddof)) if sd_vals else 0.0,
        }
        # Equal average across models of 7-point percentages
        lbls = [lbl for lbl, _ in SEVEN_POINT_SPEARMAN_SUPPORT]
        table_s50[d_key] = {
            lbl: float(np.mean([m_map[lbl] for m_map in model_seven_pt[d_key]]))
            if model_seven_pt[d_key]
            else 0.0
            for lbl in lbls
        }

    return {
        "table_s44": table_s44,
        "table_s49": table_s49,
        "table_s50": table_s50,
    }
