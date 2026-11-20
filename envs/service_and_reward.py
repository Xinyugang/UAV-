from __future__ import annotations

from typing import Dict, Sequence, Tuple
import numpy as np
from scipy.stats import spearmanr

from configs.defaults import EnvConfig, NOMINAL_TASK_PROFILES, PLATFORM_SPECS, PLATFORM_TYPES


def compute_routing_weights(
    uav_positions: np.ndarray,
    task_centers: np.ndarray,
    task_progress: np.ndarray,
    task_req: np.ndarray,
    gamma_d: float = 2.0,
    gamma_u: float = 1.0,
    d_norm: float = 100.0,
    routing_mode: str = "softmax",
    r_alt: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute query routing weights alpha_ik^t (Eq. 23) and environment service routing weights.

    Returns:
        alpha_query: shape (N, K), always computed via Eq. (23) softmax
        alpha_env: shape (N, K), equals alpha_query when routing_mode == 'softmax', or
                   uniform over unfinished tasks within 20 m when routing_mode == 'alt_20m' (S28)
        u_unfinished: shape (K,), unfinished fraction max(0, 1 - Gamma_k^t / Gamma_k^req)
    """
    dists = np.linalg.norm(
        uav_positions[:, None, :] - task_centers[None, :, :], axis=-1
    )  # shape (N, K)
    u_unfinished = np.maximum(0.0, 1.0 - task_progress / np.maximum(task_req, 1e-12))
    scores = -gamma_d * (dists / d_norm) + gamma_u * u_unfinished[None, :]
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores_shifted)
    alpha_query = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    if routing_mode == "softmax":
        alpha_env = alpha_query
    elif routing_mode == "alt_20m":
        # Supplementary Table S28: equal allocation among unfinished tasks within 20 m;
        # zero service when empty; preserves original Query-routing weights.
        elig = (dists <= r_alt) & (u_unfinished[None, :] > 0.0)
        counts = np.sum(elig, axis=-1, keepdims=True)
        alpha_env = np.where(counts > 0, elig.astype(np.float64) / np.maximum(counts, 1), 0.0)
    else:
        raise ValueError(f"Unsupported routing_mode: {routing_mode}")

    return alpha_query, alpha_env, u_unfinished


def compute_service_quality(
    uav_positions: np.ndarray,
    uav_caps: np.ndarray,
    uav_energy_j: np.ndarray,
    chi_comm: np.ndarray,
    task_centers: np.ndarray,
    task_demands: np.ndarray,
    service_rule: str = "ADD",
    beta_d: float = 0.05,
    v_max_ref: float = 8.0,
    e_max_ref_j: float = 150000.0,
) -> np.ndarray:
    """Compute service quality matrix Q_{i,k}^t of shape (N, K) (Eqs. 14-19 & Table S17).

    Boundary rules (Manuscript Eqs. 16-19):
      - Service vector x_i^svc = (v_i^max / 8, xi_i^sen, E_i / 150000, g_i * chi_i).
      - All three rules (ADD, GEO, HARM) return 0 when s_k = sum_d w_kd == 0.
      - Zero-weight components (w_kd == 0) are omitted.
      - In GEO and HARM, any zero (or <= 0) capability on a positive-weight component
        gives exact 0.0 without evaluating log(0) or 1/0, with no epsilon added to x_i^svc.
    """
    n_uav = uav_positions.shape[0]
    n_tasks = task_centers.shape[0]
    dists = np.linalg.norm(
        uav_positions[:, None, :] - task_centers[None, :, :], axis=-1
    )  # (N, K)

    v_bar = uav_caps[:, 2] / v_max_ref
    xi_sen = uav_caps[:, 4]
    e_bar = np.maximum(0.0, uav_energy_j) / e_max_ref_j
    g_chi = uav_caps[:, 5] * np.maximum(0.0, chi_comm)
    x_svc = np.stack([v_bar, xi_sen, e_bar, g_chi], axis=-1)  # (N, 4)

    rho = task_demands[:, 0]     # (K,)
    w_mat = task_demands[:, 1:5] # (K, 4)
    s_k = np.sum(w_mat, axis=-1) # (K,)
    a_ik = rho[None, :] * np.exp(-beta_d * dists)  # (N, K)

    rule = service_rule.upper()
    q_out = np.zeros((n_uav, n_tasks), dtype=np.float64)

    if rule == "ADD":
        for k in range(n_tasks):
            if s_k[k] <= 0.0:
                continue
            q_out[:, k] = a_ik[:, k] * (x_svc @ w_mat[k])
        return q_out

    if rule in ("GEO", "HARM"):
        for k in range(n_tasks):
            if s_k[k] <= 0.0:
                continue
            pos_dims = np.where(w_mat[k] > 0.0)[0]
            if pos_dims.size == 0:
                continue
            p_pos = w_mat[k, pos_dims] / s_k[k]
            x_sub = x_svc[:, pos_dims]  # (N, |pos_dims|)
            valid_uav = np.all(x_sub > 0.0, axis=-1)
            if not np.any(valid_uav):
                continue
            x_valid = x_sub[valid_uav]
            if rule == "GEO":
                val = np.exp(np.sum(p_pos[None, :] * np.log(x_valid), axis=-1))
            else:
                val = 1.0 / np.sum(p_pos[None, :] / x_valid, axis=-1)
            q_out[valid_uav, k] = a_ik[valid_uav, k] * s_k[k] * val
        return q_out

    # Supplementary Table S17 (E7 exploratory 3-component aggregation convention)
    # Uses weights w = (0.333, 0.333, 0.334) on normalized (mobility, sensing/1.5, comm_gain/1.5*chi)
    w_e7 = np.array([0.333, 0.333, 0.334], dtype=np.float64)
    m_3 = np.stack(
        [
            np.clip(v_bar, 0.0, 1.0),
            np.clip(xi_sen / 1.5, 0.0, 1.0),
            np.clip((uav_caps[:, 5] / 1.5) * chi_comm, 0.0, 1.0),
        ],
        axis=-1,
    )  # (N, 3)
    if rule == "E7_ADD":
        comp = m_3 @ w_e7
    elif rule == "E7_MIN":
        comp = 3.0 * np.min(w_e7[None, :] * m_3, axis=-1)
    elif rule == "E7_PROD":
        comp = np.prod(np.power(np.maximum(m_3, 0.0), w_e7[None, :]), axis=-1)
    else:
        raise ValueError(f"Unsupported service_rule: {service_rule}")

    for k in range(n_tasks):
        if s_k[k] <= 0.0:
            continue
        q_out[:, k] = a_ik[:, k] * s_k[k] * comp
    return q_out


def compute_step_energy(
    uav_caps: np.ndarray,
    uav_velocities: np.ndarray,
    power_matrix: np.ndarray,
    uav_positions: np.ndarray,
    task_centers: np.ndarray,
    cfg: EnvConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute per-UAV step energy consumption Delta E_i^t in Joules (Eq. 20, Table S26)."""
    masses = uav_caps[:, 0]
    xi_sen = uav_caps[:, 4]
    speed_sq = np.sum(uav_velocities ** 2, axis=-1)

    p_prop = cfg.c_prop_base + cfg.c_prop_v * (masses / cfg.m_ref_energy) * speed_sq
    p_comm = np.sum(power_matrix, axis=-1)

    dists_to_tasks = np.linalg.norm(
        uav_positions[:, None, :] - task_centers[None, :, :], axis=-1
    )
    d_min = np.min(dists_to_tasks, axis=-1)
    p_sen = cfg.c_sen * xi_sen * (d_min <= cfg.r_service).astype(np.float64)
    p_comp = np.full_like(p_prop, cfg.c_comp)

    total_power_w = p_prop + p_comm + p_sen + p_comp
    delta_e_j = cfg.dt * total_power_w
    return delta_e_j, {
        "p_prop": p_prop,
        "p_comm": p_comm,
        "p_sen": p_sen,
        "p_comp": p_comp,
    }


def compute_step_reward(
    prev_progress: np.ndarray,
    new_progress: np.ndarray,
    task_req: np.ndarray,
    delta_e_j: np.ndarray,
    n_collisions: int,
    chi_comm: np.ndarray,
    n_constraint_violations: int,
    cfg: EnvConfig,
) -> Tuple[float, Dict[str, float]]:
    """Compute shared team step reward R^t (Eq. 21 & Supplementary Table S31)."""
    # Table S31: capped incremental task progress reward
    prev_capped = np.minimum(prev_progress, task_req)
    new_capped = np.minimum(new_progress, task_req)
    r_task = float(np.sum((new_capped - prev_capped) / np.maximum(task_req, 1e-12)))

    # Normalize step energy cost relative to nominal 5-UAV hover/cruise scale (~50 J/step)
    c_energy = float(np.sum(delta_e_j) / (50.0 * max(1, delta_e_j.shape[0])))
    c_collision = float(n_collisions)
    r_comm = float(np.mean(chi_comm))
    c_constraint = float(n_constraint_violations)

    reward = (
        cfg.w1_task * r_task
        - cfg.w2_energy * c_energy
        - cfg.w3_collision * c_collision
        + cfg.w4_comm * r_comm
        - cfg.w5_constraint * c_constraint
    )
    return float(reward), {
        "r_task": r_task,
        "c_energy": c_energy,
        "c_collision": c_collision,
        "r_comm": r_comm,
        "c_constraint": c_constraint,
    }


def compute_specialization_index(
    routed_service_accum: np.ndarray, eps_log: float = 1e-8
) -> float:
    """Compute emergent specialization index S_spec in [0, 1] (Eqs. 41-42).

    Args:
        routed_service_accum: shape (N, K), accumulated sum_t alpha_ik^t * Q_{i,k}^t
    """
    n_uav, n_tasks = routed_service_accum.shape
    if n_uav <= 1 or n_tasks == 0:
        return 0.0
    log_n = float(np.log(n_uav))
    entropies = []
    for k in range(n_tasks):
        s_k = float(np.sum(routed_service_accum[:, k]))
        if s_k > 0.0:
            r_k = routed_service_accum[:, k] / s_k
        else:
            r_k = np.full(n_uav, 1.0 / n_uav, dtype=np.float64)
        h_k = -float(np.sum(r_k * np.log(r_k + eps_log)))
        entropies.append(h_k / log_n)
    return float(np.clip(1.0 - np.mean(entropies), 0.0, 1.0))


def compute_s13_contribution_alignment(
    routed_service_accum: np.ndarray,
    uav_caps: np.ndarray,
    initial_task_demands: np.ndarray,
) -> float:
    """Compute static capability contribution alignment A_ep in [0, 1] (Eqs. S5-S6, Table S13)."""
    n_uav, n_tasks = routed_service_accum.shape
    v_term = uav_caps[:, 2] / 8.0
    s_term = uav_caps[:, 4]
    e_term = uav_caps[:, 1] / 150.0  # E_max in kJ divided by 150 kJ
    g_term = uav_caps[:, 5]

    a_tasks = []
    for k in range(n_tasks):
        s_tot = float(np.sum(routed_service_accum[:, k]))
        if s_tot <= 0.0:
            a_tasks.append(0.0)
            continue
        nu_k, sig_k, eps_k, lam_k = initial_task_demands[k, 1:5]
        b_k = nu_k * v_term + sig_k * s_term + eps_k * e_term + lam_k * g_term
        b_min = float(np.min(b_k))
        b_max = float(np.max(b_k))
        if b_max - b_min <= 1e-12:
            q_k = np.full(n_uav, 0.5, dtype=np.float64)
        else:
            q_k = (b_k - b_min) / (b_max - b_min)
        r_k = routed_service_accum[:, k] / s_tot
        a_tasks.append(float(np.sum(r_k * q_k)))
    return float(np.mean(a_tasks))


def compute_c_pure_reference(
    alpha_i: np.ndarray,
    task_demands: np.ndarray,
    cand_caps: np.ndarray,
    cand_energy_j: np.ndarray,
) -> np.ndarray:
    """Compute service-derived candidate reference C_ij^ref / C_pure (Eq. S1 & S44).

    Args:
        alpha_i: shape (K,) routing weights of sender i
        task_demands: shape (K, 5) task demand matrix
        cand_caps: shape (M, 6) valid candidate capability descriptors
        cand_energy_j: shape (M,) valid candidate remaining energies in Joules
    """
    if cand_caps.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    # Weighted 4D demand mixture w_bar = sum_k alpha_ik * [nu_k, sigma_k, eps_k, lambda_k]
    w_mix = alpha_i @ task_demands[:, 1:5]  # shape (4,)
    v_j = cand_caps[:, 2] / 8.0
    xi_j = cand_caps[:, 4]
    e_j = np.maximum(0.0, cand_energy_j) / 150000.0
    g_j = cand_caps[:, 5]
    return w_mix[0] * v_j + w_mix[1] * xi_j + w_mix[2] * e_j + w_mix[3] * g_j


def evaluate_s19_counterfactual(seed: int = 271028) -> Dict[str, float]:
    """Run E9 / Table S19 512-state one-frame contribution counterfactual diagnostic (Eq. S7)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    nominal_list = [np.array(NOMINAL_TASK_PROFILES[t], dtype=np.float64) for t in ("Reconnaissance", "Sensing", "Relay")]
    cap_table = np.array([PLATFORM_SPECS[t] for t in PLATFORM_TYPES], dtype=np.float64)

    errs_v1 = []
    errs_v2 = []
    cov_v1 = 0
    cov_v2 = 0
    cov_both_and_gap = 0
    gap_only = 0
    gaps = []
    rho_v1_list = []
    rho_v2_list = []
    top1_v1 = 0
    top1_v2 = 0
    top1_both = 0

    for s_idx in range(512):
        m_cand = 2 if s_idx < 195 else 3
        task_pos = rng.uniform(10.0, 90.0, size=(3, 2))
        cand_pos = rng.uniform(5.0, 95.0, size=(m_cand, 2))
        type_idx = rng.choice(5, size=m_cand, replace=False)
        e_frac = rng.uniform(0.35, 1.0, size=m_cand)
        link_cnt = rng.integers(0, 4, size=m_cand)
        probe_k = int(rng.integers(0, 3))

        eta1 = rng.normal(0.0, 4.5, size=(m_cand, 2))
        eta2 = rng.normal(0.0, 7.5, size=(m_cand, 2))
        zeta1 = rng.normal(0.0, 0.035, size=m_cand)
        zeta2 = rng.normal(0.0, 0.060, size=m_cand)

        d_vec = nominal_list[probe_k]
        c_k = task_pos[probe_k]
        caps = cap_table[type_idx]
        e_max_kj = caps[:, 1]
        e_true_kj = e_max_kj * e_frac
        chi = link_cnt.astype(np.float64) / (3.0 + 1e-6)

        def _u_eval(pos_arr: np.ndarray, e_kj_arr: np.ndarray) -> np.ndarray:
            d_jk = np.linalg.norm(pos_arr - c_k[None, :], axis=-1)
            inner = (
                d_vec[1] * (caps[:, 2] / 8.0)
                + d_vec[2] * caps[:, 4]
                + d_vec[3] * (e_kj_arr / 150.0)
                + d_vec[4] * caps[:, 5] * chi
            )
            return 0.2 * np.exp(-0.05 * d_jk) * d_vec[0] * inner

        u_true = _u_eval(cand_pos, e_true_kj)
        u_hat1 = _u_eval(cand_pos + eta1, np.clip(e_true_kj + e_max_kj * zeta1, 0.0, e_max_kj))
        u_hat2 = _u_eval(cand_pos + eta2, np.clip(e_true_kj + e_max_kj * zeta2, 0.0, e_max_kj))

        e1 = np.abs(u_hat1 - u_true)
        e2 = np.abs(u_hat2 - u_true)
        errs_v1.extend(e1.tolist())
        errs_v2.extend(e2.tolist())

        ok1 = bool(np.max(e1) <= 0.08)
        ok2 = bool(np.max(e2) <= 0.12)
        cov_v1 += int(ok1)
        cov_v2 += int(ok2)

        u_sorted = np.sort(u_true)[::-1]
        delta_s = float(u_sorted[0] - u_sorted[1])
        gaps.append(delta_s)
        is_wide = delta_s > 0.24
        gap_only += int(is_wide)
        cov_both_and_gap += int(is_wide and ok1 and ok2)

        r1, _ = spearmanr(u_true, u_hat1)
        r2, _ = spearmanr(u_true, u_hat2)
        if np.isfinite(r1):
            rho_v1_list.append(float(r1))
        if np.isfinite(r2):
            rho_v2_list.append(float(r2))

        j_star = int(np.argmax(u_true))
        t1 = int(np.argmax(u_hat1)) == j_star
        t2 = int(np.argmax(u_hat2)) == j_star
        top1_v1 += int(t1)
        top1_v2 += int(t2)
        top1_both += int(t1 and t2)

    e1_arr = np.asarray(errs_v1, dtype=np.float64)
    e2_arr = np.asarray(errs_v2, dtype=np.float64)
    return {
        "n_states": 512,
        "n_entries": int(e1_arr.size),
        "median_err_v1": float(np.median(e1_arr)),
        "median_err_v2": float(np.median(e2_arr)),
        "p90_err_v1": float(np.quantile(e1_arr, 0.90)),
        "p90_err_v2": float(np.quantile(e2_arr, 0.90)),
        "max_err_v1": float(np.max(e1_arr)),
        "max_err_v2": float(np.max(e2_arr)),
        "cov_v1": cov_v1 / 512.0,
        "cov_v2": cov_v2 / 512.0,
        "median_gap": float(np.median(gaps)),
        "gap_only_rate": gap_only / 512.0,
        "joint_cond_rate": cov_both_and_gap / 512.0,
        "spearman_v1": float(np.mean(rho_v1_list)),
        "spearman_v2": float(np.mean(rho_v2_list)),
        "top1_v1": top1_v1 / 512.0,
        "top1_v2": top1_v2 / 512.0,
        "top1_both": top1_both / 512.0,
    }
