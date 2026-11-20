from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Sequence
import numpy as np
import torch

from configs.defaults import CAP_NORM_REF, EnvConfig
from envs.lf_uav_env import LowFidelityUAVEnv
from envs.service_and_reward import compute_routing_weights
from envs.wireless import deduplicate_and_select_candidates
from eval.eval_benchmarks import evaluate_policy_episodes
from eval.stats import paired_t_summary
from models.policy import MTCMAPPOAgent


def measure_whole_team_cpu_latency(
    agent: MTCMAPPOAgent,
    num_uavs: int = 5,
    warmup_passes: int = 1_000,
    timed_passes: int = 10_000,
    seed: int = 880_000,
) -> Dict[str, float]:
    """Measure single-thread whole-team CPU inference latency (Tables S2, S7, S14, S51c).

    Protocol:
      - Single CPU thread (`torch.set_num_threads(1)`), batch size 1 team step.
      - Includes neighbor candidate construction (`deduplicate_and_select_candidates`),
        feature normalization, network forward pass, and communication-allocation postprocessing
        (gate, softmax, power budget projection).
      - Excludes environment physics dynamics, rendering, and physical SINR evaluation.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    agent_cpu = copy.deepcopy(agent).to("cpu")
    agent_cpu.eval()

    try:
        cfg = EnvConfig(num_uavs=num_uavs)
        env = LowFidelityUAVEnv(cfg=cfg, training=False)
        env.reset(seed=seed)
        beacons = env._current_beacon_records()
        cap_ref = np.asarray(CAP_NORM_REF, dtype=np.float64)

        def _whole_team_forward_pass() -> None:
            n = num_uavs
            m_cap = cfg.wireless.m_cap
            obs_cap = cfg.obs_cap

            alpha_q, _, u_unf = compute_routing_weights(
                env.uav_pos,
                env.task_centers,
                env.task_progress,
                env.task_req,
                gamma_d=cfg.gamma_d,
                gamma_u=cfg.gamma_u,
                d_norm=cfg.d_norm,
            )
            x_self = np.concatenate(
                [
                    env.uav_pos / cfg.d_norm,
                    env.uav_vel / cfg.v_max_ref,
                    (env.uav_energy_j / cfg.e_max_ref_j)[:, None],
                ],
                axis=-1,
            ).astype(np.float32)
            z_self = (env.uav_caps / cap_ref[None, :]).astype(np.float32)
            demand_mix = (alpha_q @ env.task_demands).astype(np.float32)

            task_obs = np.zeros((n, cfg.num_tasks * 8), dtype=np.float32)
            for i in range(n):
                for k in range(cfg.num_tasks):
                    rel = (env.task_centers[k] - env.uav_pos[i]) / cfg.d_norm
                    d_ik = float(np.linalg.norm(rel))
                    dk = env.task_demands[k]
                    task_obs[i, k * 8 : (k + 1) * 8] = [
                        rel[0],
                        rel[1],
                        d_ik,
                        u_unf[k],
                        alpha_q[i, k],
                        dk[1],
                        dk[2],
                        0.5 * (dk[3] + dk[4]),
                    ]

            obs_feats = np.zeros((n, obs_cap * 4), dtype=np.float32)
            if env.obstacle_centers.shape[0] > 0:
                for i in range(n):
                    rel_all = env.obstacle_centers - env.uav_pos[i][None, :]
                    d_all = np.linalg.norm(rel_all, axis=-1)
                    order = np.argsort(d_all)[:obs_cap]
                    for s_idx, o_idx in enumerate(order):
                        obs_feats[i, s_idx * 4 : (s_idx + 1) * 4] = [
                            rel_all[o_idx, 0] / cfg.d_norm,
                            rel_all[o_idx, 1] / cfg.d_norm,
                            d_all[o_idx] / cfg.d_norm,
                            (d_all[o_idx] - cfg.d_col_obs) / 10.0,
                        ]

            nbr_x = np.zeros((n, m_cap, 5), dtype=np.float32)
            nbr_z = np.zeros((n, m_cap, 6), dtype=np.float32)
            nbr_rel = np.zeros((n, m_cap, 7), dtype=np.float32)
            nbr_dist = np.zeros((n, m_cap), dtype=np.float32)
            nbr_hist = np.zeros((n, m_cap), dtype=np.float32)
            nbr_ener = np.zeros((n, m_cap), dtype=np.float32)
            nbr_mask = np.zeros((n, m_cap), dtype=bool)
            nbr_ids = np.full((n, m_cap), -1, dtype=np.int64)

            for i in range(n):
                cand = deduplicate_and_select_candidates(
                    self_id=i,
                    self_pos=env.uav_pos[i],
                    records=beacons,
                    d_comm=cfg.wireless.d_comm,
                    m_cap=m_cap,
                )
                for slot, uid in enumerate(cand.valid_ids):
                    nbr_mask[i, slot] = True
                    nbr_ids[i, slot] = uid
                    pj = cand.padded_pos[slot]
                    vj = cand.padded_vel[slot]
                    ej = cand.padded_energy_j[slot]
                    cj = cand.padded_cap_vec[slot]
                    dc = cand.padded_dist[slot] / cfg.d_norm
                    ch = max(0.0, 1.0 - float(env.prev_sinr_linear[i, uid]) / cfg.wireless.sinr_ref)
                    ce = max(0.0, 1.0 - ej / max(cj[1] * 1000.0, 1e-6))

                    nbr_x[i, slot] = [
                        pj[0] / cfg.d_norm,
                        pj[1] / cfg.d_norm,
                        vj[0] / cfg.v_max_ref,
                        vj[1] / cfg.v_max_ref,
                        ej / cfg.e_max_ref_j,
                    ]
                    nbr_z[i, slot] = cj / cap_ref
                    nbr_dist[i, slot] = dc
                    nbr_hist[i, slot] = ch
                    nbr_ener[i, slot] = ce
                    nbr_rel[i, slot] = [
                        (pj[0] - env.uav_pos[i, 0]) / cfg.d_norm,
                        (pj[1] - env.uav_pos[i, 1]) / cfg.d_norm,
                        (vj[0] - env.uav_vel[i, 0]) / cfg.v_max_ref,
                        (vj[1] - env.uav_vel[i, 1]) / cfg.v_max_ref,
                        dc,
                        ch,
                        ce,
                    ]

            obs_dict = {
                "x_self": x_self,
                "z_self": z_self,
                "uav_roles": env.uav_roles,
                "demand_mix": demand_mix,
                "task_obs": task_obs,
                "obs_feats": obs_feats,
                "nbr_x": nbr_x,
                "nbr_z": nbr_z,
                "nbr_rel": nbr_rel,
                "nbr_dist_cost": nbr_dist,
                "nbr_hist_cost": nbr_hist,
                "nbr_energy_cost": nbr_ener,
                "nbr_mask": nbr_mask,
                "nbr_ids": nbr_ids,
            }
            agent_cpu.act(obs_dict, deterministic=True, device="cpu")

        for _ in range(warmup_passes):
            _whole_team_forward_pass()

        t_start = time.perf_counter()
        for _ in range(timed_passes):
            _whole_team_forward_pass()
        elapsed_s = time.perf_counter() - t_start

        ms_per_step = 1000.0 * elapsed_s / float(max(1, timed_passes))
        return {
            "num_uavs": float(num_uavs),
            "warmup_passes": float(warmup_passes),
            "timed_passes": float(timed_passes),
            "ms_per_team_step": ms_per_step,
        }
    finally:
        torch.set_num_threads(prev_threads)


def evaluate_team_size_scaling(
    mtc_checkpoints: Sequence[MTCMAPPOAgent],
    gat_checkpoints: Sequence[MTCMAPPOAgent],
    base_env_cfg: EnvConfig,
    team_sizes: Sequence[int] = (5, 10, 20),
    episodes_per_ckpt: int = 100,
    warmup_passes: int = 1_000,
    timed_passes: int = 10_000,
    base_seed: int = 750_000,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run Supplementary Tables S51-S52 matched team-size scaling & CPU timing comparison."""
    n_ckpts = min(len(mtc_checkpoints), len(gat_checkpoints))
    table_s51: Dict[str, Any] = {}
    table_s52: Dict[str, Any] = {}

    for n_size in team_sizes:
        cfg_n = copy.deepcopy(base_env_cfg)
        cfg_n.num_uavs = int(n_size)

        mtc_recs: List[Dict[str, float]] = []
        gat_recs: List[Dict[str, float]] = []

        for c_idx in range(n_ckpts):
            seed_blk = base_seed + 100_000 * int(n_size) + 1_000 * c_idx
            r_mtc = evaluate_policy_episodes(
                mtc_checkpoints[c_idx],
                env_cfg=cfg_n,
                num_episodes=episodes_per_ckpt,
                base_seed=seed_blk,
                device=device,
            )
            r_gat = evaluate_policy_episodes(
                gat_checkpoints[c_idx],
                env_cfg=cfg_n,
                num_episodes=episodes_per_ckpt,
                base_seed=seed_blk,
                device=device,
            )
            t_mtc = measure_whole_team_cpu_latency(
                mtc_checkpoints[c_idx],
                num_uavs=int(n_size),
                warmup_passes=warmup_passes,
                timed_passes=timed_passes,
                seed=seed_blk,
            )["ms_per_team_step"]
            t_gat = measure_whole_team_cpu_latency(
                gat_checkpoints[c_idx],
                num_uavs=int(n_size),
                warmup_passes=warmup_passes,
                timed_passes=timed_passes,
                seed=seed_blk,
            )["ms_per_team_step"]

            def _pack(res: Dict[str, Any], lat_ms: float) -> Dict[str, float]:
                eps = res["episode_records"]
                return {
                    "full_count": float(res["full_count"]),
                    "cf_full_count": float(res["cf_full_count"]),
                    "full": float(res["full_success"]),
                    "cf_full": float(res["cf_full_success"]),
                    "frames": float(res["executed_steps"]),
                    "attempts_per_ep": float(np.mean([e["attempted_links_total"] for e in eps])),
                    "effective_per_ep": float(np.mean([e["effective_links_total"] for e in eps])),
                    "attempts_per_frame": float(res["attempted_per_frame"]),
                    "effective_per_frame": float(res["effective_per_frame"]),
                    "comm": float(res["comm_success"]),
                    "time_ms": lat_ms,
                }

            mtc_recs.append(_pack(r_mtc, t_mtc))
            gat_recs.append(_pack(r_gat, t_gat))

        ddof = 1 if n_ckpts > 1 else 0

        def _summarize_method(recs: List[Dict[str, float]]) -> Dict[str, Any]:
            keys = (
                "full",
                "cf_full",
                "frames",
                "attempts_per_ep",
                "effective_per_ep",
                "attempts_per_frame",
                "effective_per_frame",
                "comm",
                "time_ms",
            )
            out_m = {
                k: {
                    "mean": float(np.mean([x[k] for x in recs])),
                    "sd": float(np.std([x[k] for x in recs], ddof=ddof)),
                }
                for k in keys
            }
            out_m["full_counts"] = [int(round(x["full_count"])) for x in recs]
            out_m["cf_full_counts"] = [int(round(x["cf_full_count"])) for x in recs]
            return out_m

        table_s51[f"N={n_size}"] = {
            "MTC": _summarize_method(mtc_recs),
            "GAT": _summarize_method(gat_recs),
        }

        # Table S52(a): MTC-minus-GAT matched-checkpoint differences before rounding
        table_s52[f"N={n_size}"] = {
            "full_diff": paired_t_summary([m["full"] - g["full"] for m, g in zip(mtc_recs, gat_recs)]),
            "cf_full_diff": paired_t_summary(
                [m["cf_full"] - g["cf_full"] for m, g in zip(mtc_recs, gat_recs)]
            ),
            "attempts_per_frame_diff": paired_t_summary(
                [m["attempts_per_frame"] - g["attempts_per_frame"] for m, g in zip(mtc_recs, gat_recs)]
            ),
            "time_ms_diff": paired_t_summary(
                [m["time_ms"] - g["time_ms"] for m, g in zip(mtc_recs, gat_recs)]
            ),
        }

    return {"table_s51": table_s51, "table_s52": table_s52}
