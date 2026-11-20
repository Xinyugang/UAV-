from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np

from configs.defaults import EnvConfig
from envs.lf_uav_env import LowFidelityUAVEnv
from eval.eval_benchmarks import evaluate_policy_episodes
from eval.stats import holm_step_down_adjust, paired_t_summary
from models.policy import MTCMAPPOAgent


def evaluate_prior_mixing_intervention(
    mtc_checkpoints: Sequence[MTCMAPPOAgent],
    gat_checkpoints: Sequence[MTCMAPPOAgent],
    env_cfg: EnvConfig,
    episodes_per_ckpt: int = 200,
    power_levels_w: Sequence[float] = (0.0, 1.0, 2.0),
    mix_alpha: float = 0.2,
    base_seed: int = 600_000,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run Supplementary Tables S40-S41 & S43 direct prior-mixing intervention.

    Modifies normalized communication prior q_j -> (1 - mix_alpha)*q_j + mix_alpha / M_valid
    before residual logits and power budget projection, with all network parameters frozen.
    """
    n_ckpts = min(len(mtc_checkpoints), len(gat_checkpoints))
    counts: Dict[float, Dict[str, List[int]]] = {}
    rates: Dict[float, Dict[str, List[float]]] = {}

    for w in power_levels_w:
        w_f = float(w)
        counts[w_f] = {"orig_mtc": [], "mixed_mtc": [], "gat": []}
        rates[w_f] = {"orig_mtc": [], "mixed_mtc": [], "gat": []}

        for idx in range(n_ckpts):
            seed_blk = base_seed + 10_000 * idx + int(round(w_f * 1000))
            res_orig = evaluate_policy_episodes(
                mtc_checkpoints[idx],
                env_cfg=env_cfg,
                num_episodes=episodes_per_ckpt,
                base_seed=seed_blk,
                ground_power_override=w_f,
                prior_uniform_mix_override=0.0,
                device=device,
            )
            res_mix = evaluate_policy_episodes(
                mtc_checkpoints[idx],
                env_cfg=env_cfg,
                num_episodes=episodes_per_ckpt,
                base_seed=seed_blk,
                ground_power_override=w_f,
                prior_uniform_mix_override=mix_alpha,
                device=device,
            )
            res_gat = evaluate_policy_episodes(
                gat_checkpoints[idx],
                env_cfg=env_cfg,
                num_episodes=episodes_per_ckpt,
                base_seed=seed_blk,
                ground_power_override=w_f,
                prior_uniform_mix_override=0.0,
                device=device,
            )
            counts[w_f]["orig_mtc"].append(res_orig["full_count"])
            counts[w_f]["mixed_mtc"].append(res_mix["full_count"])
            counts[w_f]["gat"].append(res_gat["full_count"])

            rates[w_f]["orig_mtc"].append(res_orig["full_success"])
            rates[w_f]["mixed_mtc"].append(res_mix["full_success"])
            rates[w_f]["gat"].append(res_gat["full_success"])

    # Compute the 6 paired Full contrasts in percentage points (Table S43)
    contrast_specs = []
    for w in power_levels_w:
        w_f = float(w)
        d_mo = [
            100.0 * (m - o)
            for m, o in zip(rates[w_f]["mixed_mtc"], rates[w_f]["orig_mtc"])
        ]
        contrast_specs.append((f"mixed_minus_orig_mtc_{w_f:g}W", paired_t_summary(d_mo)))
    for w in power_levels_w:
        w_f = float(w)
        d_mg = [
            100.0 * (m - g)
            for m, g in zip(rates[w_f]["mixed_mtc"], rates[w_f]["gat"])
        ]
        contrast_specs.append((f"mixed_minus_gat_{w_f:g}W", paired_t_summary(d_mg)))

    raw_ps = [spec[1]["p_sign_flip"] for spec in contrast_specs]
    holm_ps = holm_step_down_adjust(raw_ps)
    contrasts_out: Dict[str, Any] = {}
    for (name, stat_d), hp in zip(contrast_specs, holm_ps):
        stat_d["p_holm"] = hp
        contrasts_out[name] = stat_d

    summary_s41: Dict[str, Any] = {}
    for w_f, r_map in rates.items():
        ddof = 1 if n_ckpts > 1 else 0
        summary_s41[f"{w_f:g}W"] = {
            arm: {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "sd": float(np.std(vals, ddof=ddof)) if vals else 0.0,
            }
            for arm, vals in r_map.items()
        }

    return {
        "counts_s40": counts,
        "summary_s41": summary_s41,
        "contrasts_s43": contrasts_out,
    }


def evaluate_12frame_radio_window_replay(
    mtc_checkpoints: Sequence[MTCMAPPOAgent],
    gat_checkpoints: Sequence[MTCMAPPOAgent],
    env_cfg: EnvConfig,
    windows_per_model: int = 200,
    window_start_frame: int = 600,
    window_len: int = 12,
    power_levels_w: Sequence[float] = (0.0, 1.0, 2.0),
    mix_alpha: float = 0.2,
    base_seed: int = 620_000,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run Supplementary Table S42 12-frame fixed-allocation radio-window replay.

    Records valid-neighbor masks, normalized priors, residual logits, power gates, channel gains,
    and per-Watt ground-interference exposure over `window_len=12` frames, then replays original
    MTC, mixed MTC (q' = 0.8*q + 0.2/M_valid before residual), and GAT allocations across
    W in {0, 1, 2} W holding allocations fixed while varying only interference power W.
    """
    n_models = min(len(mtc_checkpoints), len(gat_checkpoints))
    eps_log = 1e-8
    p_min = env_cfg.wireless.p_min
    p_max = env_cfg.wireless.p_max
    sinr_th = env_cfg.wireless.sinr_th
    noise_w = env_cfg.wireless.n0 * env_cfg.wireless.bandwidth

    # Accumulate per-model window averages for each (W, arm)
    arms = ("MTC", "Mixed", "GAT")
    model_records: Dict[Tuple[float, str], List[Dict[str, float]]] = {
        (float(w), arm): [] for w in power_levels_w for arm in arms
    }

    for m_idx in range(n_models):
        mtc_agent = mtc_checkpoints[m_idx]
        gat_agent = gat_checkpoints[m_idx]

        # Collect `windows_per_model` 12-frame trajectory windows
        recorded_windows: List[List[Dict[str, Any]]] = []
        cfg_rec = EnvConfig(**{**env_cfg.__dict__})
        cfg_rec.max_steps = max(cfg_rec.max_steps, window_start_frame + window_len + 2)
        env = LowFidelityUAVEnv(cfg=cfg_rec, training=False)

        for w_idx in range(windows_per_model):
            obs, _ = env.reset(seed=base_seed + 10_000 * m_idx + w_idx)
            # Advance up to window_start_frame (if episode terminates early, wrap cleanly)
            for _f in range(window_start_frame):
                act_pre = mtc_agent.act(obs, deterministic=True, device=device)
                obs, _, _, done_pre, _ = env.step(act_pre["motion_cmds"], act_pre["slot_powers"])
                if done_pre:
                    obs, _ = env.reset(seed=base_seed + 900_000 + 10_000 * m_idx + w_idx)
                    break

            frames_data: List[Dict[str, Any]] = []
            for _t in range(window_len):
                mtc_out = mtc_agent.act(obs, deterministic=True, device=device)
                gat_out = gat_agent.act(obs, deterministic=True, device=device)

                # Compute fixed channel gains & unit-Watt ground exposure for this frame
                n_uav = env.cfg.num_uavs
                g_uav = env.wireless_channel.compute_channel_gain_matrix(
                    env.uav_pos, env.uav_pos, env.wireless_channel.uav_los_matrix, env.rng
                )
                np.fill_diagonal(g_uav, 0.0)

                q_cnt = env.wireless_channel.ground_positions.shape[0]
                if q_cnt > 0 and env.wireless_channel.ground_los_matrix is not None:
                    active_q = (
                        env.rng.random(q_cnt) < env.cfg.wireless.ground_active_prob
                    ).astype(np.float64)
                    g_ground = env.wireless_channel.compute_channel_gain_matrix(
                        env.wireless_channel.ground_positions,
                        env.uav_pos,
                        env.wireless_channel.ground_los_matrix,
                        env.rng,
                    )
                    unit_ground_exposure = active_q @ g_ground  # W of interference per 1 W source
                else:
                    unit_ground_exposure = np.zeros(n_uav, dtype=np.float64)

                frames_data.append(
                    {
                        "nbr_mask": obs["nbr_mask"].copy(),
                        "nbr_ids": obs["nbr_ids"].copy(),
                        "mtc_prior": mtc_out["pi_prior"].copy(),
                        "mtc_b": mtc_out["b_sampled"].numpy().copy(),
                        "mtc_gate": mtc_out["gate_mu"].copy(),
                        "gat_powers": gat_out["slot_powers"].copy(),
                        "g_uav": g_uav,
                        "unit_ground_exposure": unit_ground_exposure,
                    }
                )
                obs, _, _, done_w, _ = env.step(mtc_out["motion_cmds"], mtc_out["slot_powers"])
                if done_w and len(frames_data) < window_len:
                    obs, _ = env.reset(seed=base_seed + 1_800_000 + 10_000 * m_idx + w_idx)

            recorded_windows.append(frames_data)

        # Replay fixed allocations across interference levels W
        for w in power_levels_w:
            w_f = float(w)
            for arm in arms:
                win_entropies: List[float] = []
                win_eff_per_sf: List[float] = []
                win_receivers: List[float] = []
                win_outages: List[float] = []
                win_p10: List[float] = []
                win_p50: List[float] = []
                win_p90: List[float] = []

                for win_frames in recorded_windows:
                    sf_entropies: List[float] = []
                    n_eff_win = 0
                    n_att_win = 0
                    total_sf = 0
                    att_sinr_dbs: List[float] = []
                    n_uav = win_frames[0]["nbr_mask"].shape[0]
                    distinct_rx_per_sender = [set() for _ in range(n_uav)]

                    for fd in win_frames:
                        mask = fd["nbr_mask"]
                        ids = fd["nbr_ids"]
                        n_u, m_cap = mask.shape
                        total_sf += n_u

                        # Reconstruct slot powers for this arm (identical across W!)
                        slot_p = np.zeros((n_u, m_cap), dtype=np.float64)
                        if arm == "GAT":
                            slot_p = fd["gat_powers"].copy()
                        else:
                            for i in range(n_u):
                                v_slots = np.where(mask[i])[0]
                                m_v = v_slots.size
                                if m_v == 0:
                                    continue
                                q_v = fd["mtc_prior"][i, v_slots]
                                if arm == "Mixed":
                                    q_v = (1.0 - mix_alpha) * q_v + mix_alpha / float(m_v)
                                zeta_v = np.log(q_v + eps_log) + fd["mtc_b"][i, v_slots]
                                zeta_v = zeta_v - np.max(zeta_v)
                                shares = np.exp(zeta_v) / np.sum(np.exp(zeta_v))
                                cand_p = float(fd["mtc_gate"][i]) * p_max * shares
                                s_tot = float(np.sum(cand_p))
                                if s_tot > p_max:
                                    cand_p *= p_max / s_tot
                                slot_p[i, v_slots] = cand_p

                        # Compute normalized entropy over sender-frames with M_valid >= 2
                        for i in range(n_u):
                            v_slots = np.where(mask[i])[0]
                            if v_slots.size >= 2:
                                pv = slot_p[i, v_slots]
                                pt = float(np.sum(pv))
                                if pt > 0.0:
                                    sh = pv / pt
                                    pos = sh > 0.0
                                    h = -float(np.sum(sh[pos] * np.log(sh[pos]))) / float(
                                        np.log(v_slots.size)
                                    )
                                    sf_entropies.append(float(np.clip(h, 0.0, 1.0)))

                        # Map onto full directed matrix (N, N) and evaluate SINR at source power w_f
                        p_mat = np.zeros((n_u, n_u), dtype=np.float64)
                        for i in range(n_u):
                            for s_idx in np.where(mask[i])[0]:
                                p_mat[i, int(ids[i, s_idx])] = slot_p[i, s_idx]

                        g_uav = fd["g_uav"]
                        p_tx = np.sum(p_mat, axis=1)
                        tot_rx_j = p_tx @ g_uav
                        i_uav = np.maximum(0.0, tot_rx_j[None, :] - (p_tx[:, None] * g_uav))
                        i_gnd = w_f * fd["unit_ground_exposure"]
                        sinr_lin = (p_mat * g_uav) / (noise_w + i_uav + i_gnd[None, :])

                        for i in range(n_u):
                            for s_idx in np.where(mask[i])[0]:
                                uid = int(ids[i, s_idx])
                                if p_mat[i, uid] > p_min:
                                    n_att_win += 1
                                    s_val = float(sinr_lin[i, uid])
                                    att_sinr_dbs.append(10.0 * np.log10(max(s_val, 1e-30)))
                                    if s_val >= sinr_th:
                                        n_eff_win += 1
                                        distinct_rx_per_sender[i].add(uid)

                    if sf_entropies:
                        win_entropies.append(float(np.mean(sf_entropies)))
                    win_eff_per_sf.append(float(n_eff_win) / float(max(1, total_sf)))
                    win_receivers.append(
                        float(np.mean([len(rx_set) for rx_set in distinct_rx_per_sender]))
                    )
                    outage_w = (
                        1.0 - float(n_eff_win) / float(n_att_win) if n_att_win > 0 else 1.0
                    )
                    win_outages.append(outage_w)
                    if att_sinr_dbs:
                        q10, q50, q90 = np.quantile(att_sinr_dbs, [0.10, 0.50, 0.90])
                        win_p10.append(float(q10))
                        win_p50.append(float(q50))
                        win_p90.append(float(q90))

                model_records[(w_f, arm)].append(
                    {
                        "entropy": float(np.mean(win_entropies)) if win_entropies else 0.0,
                        "effective_links": float(np.mean(win_eff_per_sf)),
                        "receivers": float(np.mean(win_receivers)),
                        "outage": float(np.mean(win_outages)),
                        "p10_db": float(np.mean(win_p10)) if win_p10 else float("nan"),
                        "p50_db": float(np.mean(win_p50)) if win_p50 else float("nan"),
                        "p90_db": float(np.mean(win_p90)) if win_p90 else float("nan"),
                    }
                )

    summary_s42: Dict[str, Any] = {}
    for (w_f, arm), m_list in model_records.items():
        ddof = 1 if len(m_list) > 1 else 0
        key = f"{w_f:g}W_{arm}"
        summary_s42[key] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in m_list])),
                "sd": float(np.std([r[metric] for r in m_list], ddof=ddof)),
            }
            for metric in (
                "entropy",
                "effective_links",
                "receivers",
                "outage",
                "p10_db",
                "p50_db",
                "p90_db",
            )
        }
    return summary_s42


def evaluate_s30_open_loop_wireless(
    power_w: float = 2.0, seed: int = 271025
) -> Dict[str, Dict[str, float]]:
    """Run Supplementary Table S30 open-loop 192-state 12-frame wireless allocation diagnostic."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n_states = 192  # 32 one-candidate + 160 multi-candidate states
    n_frames = 12
    noise_w = 4e-15

    cand_counts = np.array([1] * 32 + [2] * 80 + [3] * 80, dtype=np.int64)
    sender_budgets = rng.uniform(0.25, 0.75, size=n_states)

    rules = ("SI-QK", "SI-Link", "SI-Mix")
    accum: Dict[str, Dict[str, List[float]]] = {
        r: {
            "entropy": [],
            "hhi": [],
            "max_share": [],
            "eff_per_frame": [],
            "distinct_partners": [],
            "coverage": [],
            "jaccard": [],
            "ang_disp": [],
            "local_svc": [],
        }
        for r in rules
    }
    pooled_att: Dict[str, int] = {r: 0 for r in rules}
    pooled_eff: Dict[str, int] = {r: 0 for r in rules}
    pooled_sinr_db: Dict[str, List[float]] = {r: [] for r in rules}

    for s in range(n_states):
        m = int(cand_counts[s])
        p_tot = float(sender_budgets[s])
        angles = rng.uniform(0.0, 2.0 * np.pi, size=m)
        base_gain = rng.uniform(0.6e-9, 2.2e-9, size=m)
        base_uav_int = rng.uniform(0.02e-9, 0.12e-9, size=m)
        base_gnd_exp = rng.uniform(0.15e-9, 0.85e-9, size=m)

        # Prescribed shares for SI-QK and SI-Link; SI-Mix = 0.8 * p_QK + 0.2 / M
        if m == 1:
            sh_qk = np.array([1.0])
            sh_lk = np.array([1.0])
        else:
            sh_qk = rng.dirichlet(np.full(m, 3.5))
            sh_lk = rng.dirichlet(np.linspace(4.0, 1.2, m))
        sh_mix = 0.8 * sh_qk + 0.2 / float(m)
        share_map = {"SI-QK": sh_qk, "SI-Link": sh_lk, "SI-Mix": sh_mix}

        # Shared channel realizations across all 3 rules (12 frames)
        ln_des = np.exp(rng.normal(0.0, 0.35, size=(n_frames, m)) - 0.5 * 0.35 ** 2)
        ln_uav = np.exp(rng.normal(0.0, 0.20, size=(n_frames, m)) - 0.5 * 0.20 ** 2)
        z_c = rng.normal(0.0, 1.0, size=(n_frames, 1))
        z_j = rng.normal(0.0, 1.0, size=(n_frames, m))
        ln_gnd = np.exp(
            0.3 * (0.7 * z_c + 0.3 * z_j) - 0.5 * (0.3 ** 2) * (0.7 ** 2 + 0.3 ** 2)
        )

        g_tf = base_gain[None, :] * ln_des
        i_uav_tf = base_uav_int[None, :] * ln_uav
        i_gnd_tf = power_w * base_gnd_exp[None, :] * ln_gnd

        for r_name, sh in share_map.items():
            p_j = p_tot * sh
            if m >= 2:
                pos = sh > 0.0
                h = -float(np.sum(sh[pos] * np.log(sh[pos]))) / float(np.log(m))
                accum[r_name]["entropy"].append(h)
            accum[r_name]["hhi"].append(float(np.sum(sh ** 2)))
            accum[r_name]["max_share"].append(float(np.max(sh)))
            ang_d = 1.0 - float(np.abs(np.sum(sh * np.exp(1j * angles))))
            accum[r_name]["ang_disp"].append(ang_d)

            att_mask = p_j > 0.02
            sinr_tf = (p_j[None, :] * g_tf) / (i_uav_tf + i_gnd_tf + noise_w)
            eff_tf = att_mask[None, :] & (sinr_tf >= 1.0)

            n_att_s = int(np.sum(att_mask)) * n_frames
            n_eff_s = int(np.sum(eff_tf))
            pooled_att[r_name] += n_att_s
            pooled_eff[r_name] += n_eff_s
            if np.any(att_mask):
                vals_db = 10.0 * np.log10(np.maximum(sinr_tf[:, att_mask], 1e-30))
                pooled_sinr_db[r_name].extend(vals_db.ravel().tolist())

            accum[r_name]["eff_per_frame"].append(float(n_eff_s) / float(n_frames))
            distinct_cnt = int(np.sum(np.any(eff_tf, axis=0)))
            accum[r_name]["distinct_partners"].append(float(distinct_cnt))
            accum[r_name]["coverage"].append(float(distinct_cnt) / float(m))

            jacc_list = []
            for t in range(n_frames - 1):
                u_cnt = int(np.sum(eff_tf[t] | eff_tf[t + 1]))
                if u_cnt > 0:
                    i_cnt = int(np.sum(eff_tf[t] & eff_tf[t + 1]))
                    jacc_list.append(float(i_cnt) / float(u_cnt))
            if jacc_list:
                accum[r_name]["jaccard"].append(float(np.mean(jacc_list)))

            # Local service increment over 12 frames (0.2 s each)
            chi_t = np.sum(eff_tf, axis=-1) / (float(m) + 1e-6)
            svc_inc = float(np.sum(0.2 * (0.32 + 0.08 * chi_t)))
            accum[r_name]["local_svc"].append(svc_inc)

    out: Dict[str, Dict[str, float]] = {}
    for r_name in rules:
        dbs = np.asarray(pooled_sinr_db[r_name], dtype=np.float64)
        q10, q25, q50, q75, q90 = (
            np.quantile(dbs, [0.10, 0.25, 0.50, 0.75, 0.90])
            if dbs.size > 0
            else (-50.0, -40.0, -20.0, -10.0, 0.0)
        )
        out[r_name] = {
            "normalized_entropy": float(np.mean(accum[r_name]["entropy"])),
            "hhi": float(np.mean(accum[r_name]["hhi"])),
            "max_share": float(np.mean(accum[r_name]["max_share"])),
            "effective_neighbors_per_frame": float(np.mean(accum[r_name]["eff_per_frame"])),
            "distinct_partners_per_window": float(np.mean(accum[r_name]["distinct_partners"])),
            "partner_coverage": float(np.mean(accum[r_name]["coverage"])),
            "consecutive_jaccard": float(np.mean(accum[r_name]["jaccard"]))
            if accum[r_name]["jaccard"]
            else 0.0,
            "angular_dispersion": float(np.mean(accum[r_name]["ang_disp"])),
            "outage_fraction": 1.0 - float(pooled_eff[r_name]) / float(max(1, pooled_att[r_name])),
            "sinr_p10_db": float(q10),
            "sinr_p25_db": float(q25),
            "sinr_p50_db": float(q50),
            "sinr_p75_db": float(q75),
            "sinr_p90_db": float(q90),
            "local_service_increment": float(np.mean(accum[r_name]["local_svc"])),
        }
    return out


def evaluate_s33_asymmetric_exposure(seed: int = 271026) -> Dict[str, Dict[str, float]]:
    """Run Supplementary Table S33 controlled capability-feasibility conflict (Eq. S9)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n_states = 200
    n_frames = 20
    caps = np.array([1.0, 0.55, 0.45], dtype=np.float64)
    nom_gains = np.array([1.6e-9, 1.3e-9, 1.1e-9], dtype=np.float64)
    bg_powers = np.array([0.08e-9, 0.09e-9, 0.08e-9], dtype=np.float64)
    nom_exp = np.array([0.90e-9, 0.055e-9, 0.035e-9], dtype=np.float64)

    # Task-prior favors capable candidate 1; Link-prior is balanced
    p_task = rng.dirichlet([24.0, 4.5, 3.0], size=n_states)
    p_link = rng.dirichlet([18.0, 17.0, 15.0], size=n_states)
    p_mix = 0.8 * p_task + 0.2 / 3.0

    g_state = np.exp(rng.normal(0.0, 0.20, size=(n_states, 1, 3)) - 0.5 * 0.20 ** 2)
    x_state = np.exp(rng.normal(0.0, 0.25, size=(n_states, 1, 3)) - 0.5 * 0.25 ** 2)
    g_frame = np.exp(rng.normal(0.0, 0.25, size=(n_states, n_frames, 3)) - 0.5 * 0.25 ** 2)
    x_frame = np.exp(rng.normal(0.0, 0.20, size=(n_states, n_frames, 3)) - 0.5 * 0.20 ** 2)

    g_all = nom_gains[None, None, :] * g_state * g_frame
    x_all = nom_exp[None, None, :] * x_state * x_frame

    results: Dict[str, Dict[str, float]] = {}
    for w in (0.0, 1.0, 2.0):
        for name, shares in (("Task-prior", p_task), ("Link-prior", p_link), ("Mixed", p_mix)):
            att = shares > 0.02  # (200, 3) with 1 W budget
            sinr = (shares[:, None, :] * g_all) / (bg_powers[None, None, :] + w * x_all)
            eff = att[:, None, :] & (sinr >= 1.0)

            # Eq. (S9): frame-averaged capability-weighted feasible priority U
            u_states = np.mean(np.sum(shares[:, None, :] * caps[None, None, :] * eff, axis=-1), axis=-1)
            eff_nbrs = float(np.mean(np.sum(eff, axis=-1)))
            ent = float(np.mean(-np.sum(shares * np.log(np.maximum(shares, 1e-12)), axis=-1) / np.log(3.0)))
            outage = 1.0 - float(np.sum(eff)) / float(max(1, int(np.sum(att)) * n_frames))

            results[f"{w:g}W_{name}"] = {
                "utility": float(np.mean(u_states)),
                "effective_neighbors": eff_nbrs,
                "entropy": ent,
                "outage": outage,
            }
    return results


def evaluate_s36_exposure_rotation(
    seed: int = 271029, bootstrap_seed: int = 271030, n_boot: int = 2000
) -> Dict[str, Any]:
    """Run Supplementary Table S36 exposure-location rotation intervention & DiD test."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n_states = 256
    n_frames = 12

    p_cap = rng.dirichlet([37.5, 7.5, 5.0], size=n_states)
    p_bal = rng.dirichlet([28.8, 27.2, 24.0], size=n_states)
    p_mix = 0.8 * p_cap + 0.2 / 3.0

    nom_caps = np.array([1.0, 0.55, 0.45], dtype=np.float64)
    cap_mult = np.exp(rng.normal(0.0, 0.06, size=(n_states, 3)) - 0.5 * 0.06 ** 2)
    caps = np.minimum(1.0, nom_caps[None, :] * cap_mult)

    nom_gains = np.array([1.6e-9, 1.3e-9, 1.1e-9], dtype=np.float64)
    bg_powers = np.array([0.08e-9, 0.09e-9, 0.08e-9], dtype=np.float64)
    nom_exp = np.array([0.90e-9, 0.055e-9, 0.035e-9], dtype=np.float64)

    g_st = np.exp(rng.normal(0.0, 0.20, size=(n_states, 1, 3)) - 0.5 * 0.20 ** 2)
    g_fr = np.exp(rng.normal(0.0, 0.25, size=(n_states, n_frames, 3)) - 0.5 * 0.25 ** 2)
    x_st = np.exp(rng.normal(0.0, 0.25, size=(n_states, 1, 3)) - 0.5 * 0.25 ** 2)
    x_fr = np.exp(rng.normal(0.0, 0.20, size=(n_states, n_frames, 3)) - 0.5 * 0.20 ** 2)

    g_all = nom_gains[None, None, :] * g_st * g_fr
    x_aligned = nom_exp[None, None, :] * x_st * x_fr
    # Cyclically shift exposure vector by 1 candidate position (candidate 1 -> candidate 2)
    x_rotated = np.roll(x_aligned, shift=1, axis=-1)

    def _eval_arm(shares: np.ndarray, q_w: float, x_mat: np.ndarray) -> Tuple[np.ndarray, float, float]:
        att = shares > 0.02
        sinr = (shares[:, None, :] * g_all) / (bg_powers[None, None, :] + q_w * x_mat)
        eff = att[:, None, :] & (sinr >= 1.0)
        u_vec = np.mean(np.sum(shares[:, None, :] * caps[:, None, :] * eff, axis=-1), axis=-1)
        eff_n = float(np.mean(np.sum(eff, axis=-1)))
        ent = float(np.mean(-np.sum(shares * np.log(np.maximum(shares, 1e-12)), axis=-1) / np.log(3.0)))
        return u_vec, eff_n, ent

    regimes = [
        ("0W_reference", 0.0, x_aligned),
        ("2W_aligned", 2.0, x_aligned),
        ("2W_rotated", 2.0, x_rotated),
    ]
    boot_rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    boot_idx = boot_rng.integers(0, n_states, size=(n_boot, n_states))

    rows: Dict[str, Any] = {}
    u_store: Dict[str, np.ndarray] = {}
    for reg_lbl, q_w, x_mat in regimes:
        for r_lbl, sh in (("Capability", p_cap), ("Balanced", p_bal), ("Mixed", p_mix)):
            u_vec, eff_n, ent = _eval_arm(sh, q_w, x_mat)
            key = f"{reg_lbl}_{r_lbl}"
            u_store[key] = u_vec
            b_means = np.mean(u_vec[boot_idx], axis=-1)
            ci_lo, ci_hi = np.quantile(b_means, [0.025, 0.975])
            rows[key] = {
                "utility_mean": float(np.mean(u_vec)),
                "ci_low": float(ci_lo),
                "ci_high": float(ci_hi),
                "effective_neighbors": eff_n,
                "entropy": ent,
            }

    diff_aligned = u_store["2W_aligned_Capability"] - u_store["2W_aligned_Balanced"]
    diff_rotated = u_store["2W_rotated_Capability"] - u_store["2W_rotated_Balanced"]
    did_vec = diff_aligned - diff_rotated
    did_boot = np.mean(did_vec[boot_idx], axis=-1)

    return {
        "rows": rows,
        "cap_minus_bal_2W_aligned": float(np.mean(diff_aligned)),
        "cap_minus_bal_2W_rotated": float(np.mean(diff_rotated)),
        "did_aligned_minus_rotated": float(np.mean(did_vec)),
        "did_ci_95": [float(np.quantile(did_boot, 0.025)), float(np.quantile(did_boot, 0.975))],
    }
