from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from configs.defaults import EnvConfig, ModelConfig, WirelessConfig
from envs.service_and_reward import evaluate_s19_counterfactual
from eval.eval_benchmarks import evaluate_policy_episodes, evaluate_s6_allocation_diagnostics
from eval.eval_interference import (
    evaluate_12frame_radio_window_replay,
    evaluate_prior_mixing_intervention,
    evaluate_s30_open_loop_wireless,
    evaluate_s33_asymmetric_exposure,
    evaluate_s36_exposure_rotation,
)
from eval.eval_masking import evaluate_distance_matched_masking, evaluate_lf_attention_masking
from eval.eval_pybullet import (
    evaluate_pybullet_transfer_and_sweeps,
    evaluate_s53_single_platform_suite,
    evaluate_s54_closed_loop_compensation,
)
from eval.eval_scaling_timing import evaluate_team_size_scaling, measure_whole_team_cpu_latency
from eval.eval_spearman import evaluate_s44_candidate_correlations
from models.policy import B2LocalPowerHeuristic, B3VelocitySafetyFilter, MTCMAPPOAgent
from train.mappo_trainer import load_checkpoint


def _load_or_init(paths: List[str], method: str, device: str) -> List[MTCMAPPOAgent]:
    if paths:
        return [load_checkpoint(p, device=device)[0] for p in paths]
    agent = MTCMAPPOAgent(ModelConfig(method=method)).to(device)
    agent.eval()
    return [agent]


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation & diagnostic suite for MTC-MAPPO.")
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        choices=[
            "benchmark",
            "s6_alloc",
            "masking_lf",
            "masking_dist_matched",
            "prior_mixing",
            "radio_window_s42",
            "open_loop_wireless",
            "spearman_s44",
            "scaling_s51",
            "cpu_timing",
            "pybullet_transfer",
            "pybullet_s53_single",
            "pybullet_s54_comp",
        ],
    )
    parser.add_argument("--mtc-ckpts", type=str, nargs="*", default=[])
    parser.add_argument("--gat-ckpts", type=str, nargs="*", default=[])
    parser.add_argument("--comparator", type=str, default="none", choices=["none", "b2", "b3"])
    parser.add_argument("--wireless-mode", type=str, default="e1", choices=["lf", "e1"])
    parser.add_argument("--service-rule", type=str, default="ADD")
    parser.add_argument("--routing-mode", type=str, default="softmax", choices=["softmax", "alt_20m"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-ep-steps", type=int, default=1500)
    parser.add_argument("--team-sizes", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--warmup-passes", type=int, default=100)
    parser.add_argument("--timed-passes", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=str, default="outputs/eval_result.json")
    args = parser.parse_args()

    env_cfg = EnvConfig(
        max_steps=args.max_ep_steps,
        service_rule=args.service_rule,
        routing_mode=args.routing_mode,
        wireless=WirelessConfig(wireless_mode=args.wireless_mode),
    )

    mtc_agents = _load_or_init(args.mtc_ckpts, "mtc_mappo", args.device)
    gat_agents = _load_or_init(args.gat_ckpts, "gat_mappo", args.device)

    if args.suite == "benchmark":
        if args.comparator == "b2":
            actor_obj = B2LocalPowerHeuristic(mtc_agents[0])
        elif args.comparator == "b3":
            actor_obj = B3VelocitySafetyFilter(mtc_agents[0])
        else:
            actor_obj = mtc_agents[0]
        res = evaluate_policy_episodes(
            actor_obj, env_cfg=env_cfg, num_episodes=args.episodes, device=args.device
        )
        res.pop("episode_records", None)
    elif args.suite == "s6_alloc":
        res = evaluate_s6_allocation_diagnostics(
            mtc_agents[0], env_cfg=env_cfg, num_episodes=args.episodes, device=args.device
        )
    elif args.suite == "masking_lf":
        res = evaluate_lf_attention_masking(
            mtc_agents[0], env_cfg=env_cfg, num_episodes=args.episodes, device=args.device
        )
    elif args.suite == "masking_dist_matched":
        res = evaluate_distance_matched_masking(
            mtc_agents, env_cfg=env_cfg, episodes_per_ckpt=args.episodes, device=args.device
        )
    elif args.suite == "prior_mixing":
        res = evaluate_prior_mixing_intervention(
            mtc_agents,
            gat_agents,
            env_cfg=env_cfg,
            episodes_per_ckpt=args.episodes,
            device=args.device,
        )
    elif args.suite == "radio_window_s42":
        res = evaluate_12frame_radio_window_replay(
            mtc_agents,
            gat_agents,
            env_cfg=env_cfg,
            windows_per_model=args.episodes,
            window_start_frame=min(600, max(0, args.max_ep_steps - 15)),
            device=args.device,
        )
    elif args.suite == "open_loop_wireless":
        res = {
            "s19_counterfactual": evaluate_s19_counterfactual(),
            "s30_open_loop": evaluate_s30_open_loop_wireless(),
            "s33_asymmetric_exposure": evaluate_s33_asymmetric_exposure(),
            "s36_exposure_rotation": evaluate_s36_exposure_rotation(),
        }
    elif args.suite == "spearman_s44":
        res = evaluate_s44_candidate_correlations(
            mtc_agents,
            env_cfg=env_cfg,
            states_per_model=max(10, args.episodes),
            behavior_states_per_model=max(4, args.episodes // 3),
            behavior_horizon=min(50, args.max_ep_steps),
            device=args.device,
        )
    elif args.suite == "scaling_s51":
        res = evaluate_team_size_scaling(
            mtc_agents,
            gat_agents,
            base_env_cfg=env_cfg,
            team_sizes=args.team_sizes,
            episodes_per_ckpt=args.episodes,
            warmup_passes=args.warmup_passes,
            timed_passes=args.timed_passes,
            device=args.device,
        )
    elif args.suite == "cpu_timing":
        res = {
            f"N={n}": measure_whole_team_cpu_latency(
                mtc_agents[0],
                num_uavs=n,
                warmup_passes=args.warmup_passes,
                timed_passes=args.timed_passes,
            )
            for n in args.team_sizes
        }
    elif args.suite == "pybullet_transfer":
        res = evaluate_pybullet_transfer_and_sweeps(
            {"mtc_mappo": mtc_agents, "gat_mappo": gat_agents},
            env_cfg=env_cfg,
            episodes_per_ckpt=args.episodes,
            sweep_methods=("gat_mappo", "mtc_mappo"),
            device=args.device,
        )
    elif args.suite == "pybullet_s53_single":
        res = evaluate_s53_single_platform_suite(num_instances_per_type=args.episodes)
    elif args.suite == "pybullet_s54_comp":
        res = evaluate_s54_closed_loop_compensation(
            mtc_agents, env_cfg=env_cfg, episodes_per_cell=args.episodes, device=args.device
        )
    else:
        raise ValueError(f"Unhandled suite: {args.suite}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved evaluation results ({args.suite}) to {out_path}")


if __name__ == "__main__":
    main()
