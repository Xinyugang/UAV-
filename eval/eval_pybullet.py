from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Sequence
import numpy as np

from configs.defaults import (
    CompensatorConfig,
    EnvConfig,
    PLATFORM_TYPES,
    PyBulletConfig,
)
from envs.pybullet_uav_env import PyBulletUAVEnv, run_single_platform_tracking_trial
from eval.stats import paired_t_summary
from models.policy import MTCMAPPOAgent


def _run_pybullet_ckpt_condition(
    agent: MTCMAPPOAgent,
    env_cfg: EnvConfig,
    pb_cfg: PyBulletConfig,
    num_episodes: int,
    base_seed: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    pb_env = PyBulletUAVEnv(env_cfg=env_cfg, pb_cfg=pb_cfg)
    try:
        recs: List[Dict[str, float]] = []
        for ep in range(num_episodes):
            obs, _ = pb_env.reset(seed=base_seed + ep)
            done = False
            info: Dict[str, Any] = {}
            while not done:
                act_out = agent.act(obs, deterministic=True, device=device)
                obs, _, _, done, info = pb_env.step(
                    act_out["motion_cmds"], act_out["slot_powers"]
                )
            recs.append(info["episode_metrics"])

        full_cnt = int(round(sum(r["full_success"] for r in recs)))
        cf_cnt = int(round(sum(r["cf_full_success"] for r in recs)))

        def _avg(k: str) -> float:
            vals = [r[k] for r in recs if np.isfinite(r[k])]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "full_count": full_cnt,
            "cf_full_count": cf_cnt,
            "full_rate": full_cnt / float(max(1, num_episodes)),
            "cf_full_rate": cf_cnt / float(max(1, num_episodes)),
            "comm_success": _avg("comm_success"),
            "mean_sinr_db": _avg("mean_sinr_db"),
            "motor_energy_index": _avg("motor_energy_index"),
            "tracking_rmse": _avg("tracking_rmse"),
            "saturation_rate": _avg("saturation_rate"),
            "utp": _avg("utp"),
            "prediction_rmse": _avg("prediction_rmse"),
            "estimation_rmse_vs_r0": _avg("estimation_rmse_vs_r0"),
            "extra_comp_time_ms": _avg("extra_comp_time_ms"),
        }
    finally:
        pb_env.close()


def evaluate_pybullet_transfer_and_sweeps(
    method_checkpoints: Mapping[str, Sequence[MTCMAPPOAgent]],
    env_cfg: EnvConfig,
    episodes_per_ckpt: int = 100,
    delay_levels_ms: Sequence[float] = (0.0, 40.0, 80.0, 120.0, 160.0),
    noise_levels: Sequence[float] = (0.00, 0.05, 0.10, 0.15, 0.20),
    attitude_levels: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    sweep_methods: Sequence[str] = ("mappo", "mappo_morph", "gat_mappo", "mtc_mappo"),
    base_seed: int = 800_000,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run E8 Nominal PyBullet Transfer (Table 11, S15, S18) & 1-Factor Sweeps (Tables S23-S24).

    Retention ratios average unrounded within-checkpoint count ratios k_s(delta) / k_s(0)
    per Supplementary Table S23-S24.
    """
    nominal_summary: Dict[str, Any] = {}
    nominal_counts_by_method: Dict[str, List[int]] = {}

    for method_name, ckpts in method_checkpoints.items():
        ckpt_recs: List[Dict[str, Any]] = []
        for s_idx, agent in enumerate(ckpts):
            pb_nom = PyBulletConfig(gui=False)
            res = _run_pybullet_ckpt_condition(
                agent,
                env_cfg=env_cfg,
                pb_cfg=pb_nom,
                num_episodes=episodes_per_ckpt,
                base_seed=base_seed + 10_000 * s_idx,
                device=device,
            )
            ckpt_recs.append(res)

        nominal_counts_by_method[method_name] = [int(r["full_count"]) for r in ckpt_recs]
        ddof = 1 if len(ckpt_recs) > 1 else 0
        nominal_summary[method_name] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in ckpt_recs])),
                "sd": float(np.std([r[metric] for r in ckpt_recs], ddof=ddof)),
            }
            for metric in (
                "full_rate",
                "cf_full_rate",
                "comm_success",
                "mean_sinr_db",
                "motor_energy_index",
                "tracking_rmse",
                "saturation_rate",
                "utp",
            )
        }
        nominal_summary[method_name]["per_checkpoint"] = ckpt_recs

    # One-factor sweeps with count-based retention R(delta) = (1/S) * sum_s k_s(delta) / k_s(0)
    retention_s23: Dict[str, Dict[str, float]] = {}
    counts_s24: Dict[str, Dict[str, List[str]]] = {}

    for method_name in sweep_methods:
        if method_name not in method_checkpoints:
            continue
        ckpts = method_checkpoints[method_name]
        nom_counts = nominal_counts_by_method[method_name]
        retention_s23[method_name] = {}
        counts_s24[method_name] = {}

        def _eval_factor(factor_lbl: str, levels: Sequence[float]) -> None:
            for lvl in levels:
                lvl_f = float(lvl)
                key = f"{factor_lbl}={lvl_f:g}"
                if lvl_f == 0.0:
                    cur_counts = list(nom_counts)
                else:
                    cur_counts = []
                    for s_idx, agent in enumerate(ckpts):
                        pb_c = PyBulletConfig(gui=False)
                        if factor_lbl == "delay_ms":
                            pb_c.control_delay_ms = lvl_f
                        elif factor_lbl == "comm_noise":
                            pb_c.comm_noise_std = lvl_f
                        elif factor_lbl == "attitude_kappa":
                            pb_c.attitude_atten_kappa = lvl_f
                        r_d = _run_pybullet_ckpt_condition(
                            agent,
                            env_cfg=env_cfg,
                            pb_cfg=pb_c,
                            num_episodes=episodes_per_ckpt,
                            base_seed=base_seed + 10_000 * s_idx + int(round(lvl_f * 100)),
                            device=device,
                        )
                        cur_counts.append(int(r_d["full_count"]))

                ratios = [
                    float(kc) / float(max(1, k0)) if k0 > 0 else 0.0
                    for kc, k0 in zip(cur_counts, nom_counts)
                ]
                retention_s23[method_name][key] = float(np.mean(ratios)) if ratios else 0.0
                counts_s24[method_name][key] = [
                    f"{kc}/{k0}" for kc, k0 in zip(cur_counts, nom_counts)
                ]

        _eval_factor("delay_ms", delay_levels_ms)
        _eval_factor("comm_noise", noise_levels)
        _eval_factor("attitude_kappa", attitude_levels)

    return {
        "nominal_table11": nominal_summary,
        "retention_s23": retention_s23,
        "counts_s24": counts_s24,
    }


def evaluate_s53_single_platform_suite(
    num_instances_per_type: int = 20,
    total_duration_s: float = 150.0,
    dist_start_s: float = 30.0,
    dist_end_s: float = 90.0,
    recovery_hold_s: float = 5.0,
    base_seed: int = 910_000,
) -> Dict[str, Any]:
    """Run Section 7.3 / Supplementary Table S53 single-platform tracking & recovery suite."""
    summary: Dict[str, Any] = {}
    for t_idx, p_type in enumerate(PLATFORM_TYPES):
        for ctrl_grp, ctrl_lbl in (("A", "PID"), ("C", "PID+PC")):
            trials = [
                run_single_platform_tracking_trial(
                    platform_type=p_type,
                    controller_group=ctrl_grp,
                    seed=base_seed + 1000 * t_idx + inst,
                    total_duration_s=total_duration_s,
                    dist_start_s=dist_start_s,
                    dist_end_s=dist_end_s,
                    recovery_hold_s=recovery_hold_s,
                )
                for inst in range(num_instances_per_type)
            ]
            rmses = [tr["rmse_mps"] for tr in trials]
            max_errs = [tr["max_err_mps"] for tr in trials]
            sats = [tr["saturation_fraction"] for tr in trials]
            rec_times = [tr["recovery_time_s"] for tr in trials if tr["recovered"]]
            not_rec = sum(1 for tr in trials if not tr["recovered"])
            ddof = 1 if len(trials) > 1 else 0

            if rec_times:
                q25, q50, q75 = np.quantile(rec_times, [0.25, 0.50, 0.75])
            else:
                q25 = q50 = q75 = float("nan")

            summary[f"{p_type}_{ctrl_lbl}"] = {
                "platform_type": p_type,
                "controller": ctrl_lbl,
                "n_tests": len(trials),
                "rmse_mean": float(np.mean(rmses)),
                "rmse_sd": float(np.std(rmses, ddof=ddof)),
                "max_err_mean": float(np.mean(max_errs)),
                "max_err_sd": float(np.std(max_errs, ddof=ddof)),
                "recovery_median_s": float(q50),
                "recovery_iqr_s": [float(q25), float(q75)],
                "not_recovered_count": int(not_rec),
                "saturation_mean": float(np.mean(sats)),
                "saturation_sd": float(np.std(sats, ddof=ddof)),
            }
    return summary


def evaluate_s54_closed_loop_compensation(
    mtc_checkpoints: Sequence[MTCMAPPOAgent],
    env_cfg: EnvConfig,
    episodes_per_cell: int = 100,
    base_seed: int = 950_000,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run Section 7.4 / Tables 12 & S54-S56 frozen-policy A/B/C execution compensation study.

    - Evaluates Controller Groups A (PID), B (PID + prediction FF), C (PID + pred FF + residual)
      across 'nominal', 'low_error', and 'high_error' conditions.
    - Group A shares its disturbed run set across 'low_error' and 'high_error' because Group A
      does not consume predictor inputs (Supplementary p. 45, S54).
    """
    n_ckpts = len(mtc_checkpoints)
    cell_ckpt_records: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    # Shared disturbed Group A cache so low_error A and high_error A reuse identical runs
    disturbed_a_cache: List[Dict[str, Any]] = []
    for s_idx, agent in enumerate(mtc_checkpoints):
        pb_dist_a = PyBulletConfig(
            gui=False,
            compensator=CompensatorConfig(group="A", disturbance_mode="low_error"),
        )
        disturbed_a_cache.append(
            _run_pybullet_ckpt_condition(
                agent,
                env_cfg=copy.deepcopy(env_cfg),
                pb_cfg=pb_dist_a,
                num_episodes=episodes_per_cell,
                base_seed=base_seed + 10_000 * s_idx,
                device=device,
            )
        )

    for cond in ("nominal", "low_error", "high_error"):
        for grp in ("A", "B", "C"):
            if grp == "A" and cond in ("low_error", "high_error"):
                cell_ckpt_records[(cond, grp)] = disturbed_a_cache
                continue
            recs_cell: List[Dict[str, Any]] = []
            for s_idx, agent in enumerate(mtc_checkpoints):
                pb_cell = PyBulletConfig(
                    gui=False,
                    compensator=CompensatorConfig(group=grp, disturbance_mode=cond),
                )
                recs_cell.append(
                    _run_pybullet_ckpt_condition(
                        agent,
                        env_cfg=copy.deepcopy(env_cfg),
                        pb_cfg=pb_cell,
                        num_episodes=episodes_per_cell,
                        base_seed=base_seed + 10_000 * s_idx,
                        device=device,
                    )
                )
            cell_ckpt_records[(cond, grp)] = recs_cell

    ddof = 1 if n_ckpts > 1 else 0
    table_s54: Dict[str, Any] = {}
    for (cond, grp), c_list in cell_ckpt_records.items():
        key = f"{cond}_{grp}"
        table_s54[key] = {
            metric: {
                "mean": float(np.mean([x[metric] for x in c_list])),
                "sd": float(np.std([x[metric] for x in c_list], ddof=ddof)),
            }
            for metric in (
                "full_rate",
                "cf_full_rate",
                "tracking_rmse",
                "saturation_rate",
                "prediction_rmse",
                "estimation_rmse_vs_r0",
                "motor_energy_index",
                "extra_comp_time_ms",
            )
        }
        table_s54[key]["full_counts"] = [int(x["full_count"]) for x in c_list]
        table_s54[key]["cf_full_counts"] = [int(x["cf_full_count"]) for x in c_list]

    # Table 12 / Table S55 within-checkpoint paired contrasts
    def _contrast(cond_name: str, grp1: str, grp2: str) -> Dict[str, Any]:
        r1 = cell_ckpt_records[(cond_name, grp1)]
        r2 = cell_ckpt_records[(cond_name, grp2)]
        return {
            "full_diff": paired_t_summary([a["full_rate"] - b["full_rate"] for a, b in zip(r1, r2)]),
            "cf_full_diff": paired_t_summary(
                [a["cf_full_rate"] - b["cf_full_rate"] for a, b in zip(r1, r2)]
            ),
            "tracking_rmse_diff": paired_t_summary(
                [a["tracking_rmse"] - b["tracking_rmse"] for a, b in zip(r1, r2)]
            ),
        }

    table_s55 = {
        "high_error_C_minus_B": _contrast("high_error", "C", "B"),
        "high_error_C_minus_A": _contrast("high_error", "C", "A"),
        "nominal_C_minus_A": _contrast("nominal", "C", "A"),
    }
    return {"table_s54": table_s54, "table_s55_table12": table_s55}
