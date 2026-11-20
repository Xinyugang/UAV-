from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from configs.defaults import EnvConfig, ModelConfig, PPOConfig, WirelessConfig
from envs.lf_uav_env import LowFidelityUAVEnv
from train.mappo_trainer import CTDEMAPPOTrainer, save_checkpoint


# Supplementary Table S48: One-Factor-at-a-Time (OFAT) matching-score parameter settings
# Default: tau_a = 0.5, (alpha_d, alpha_c, alpha_e) = (0.5, 0.4, 0.3)
OFAT_PARAMETER_GRID: List[Tuple[str, Dict[str, float]]] = [
    ("default",       {"tau_a": 0.50, "alpha_d": 0.50, "alpha_c": 0.40, "alpha_e": 0.30}),
    ("tau_a=0.25",    {"tau_a": 0.25, "alpha_d": 0.50, "alpha_c": 0.40, "alpha_e": 0.30}),
    ("tau_a=1.0",     {"tau_a": 1.00, "alpha_d": 0.50, "alpha_c": 0.40, "alpha_e": 0.30}),
    ("alpha_d=0.25",  {"tau_a": 0.50, "alpha_d": 0.25, "alpha_c": 0.40, "alpha_e": 0.30}),
    ("alpha_d=1.0",   {"tau_a": 0.50, "alpha_d": 1.00, "alpha_c": 0.40, "alpha_e": 0.30}),
    ("alpha_c=0.2",   {"tau_a": 0.50, "alpha_d": 0.50, "alpha_c": 0.20, "alpha_e": 0.30}),
    ("alpha_c=0.8",   {"tau_a": 0.50, "alpha_d": 0.50, "alpha_c": 0.80, "alpha_e": 0.30}),
    ("alpha_e=0.15",  {"tau_a": 0.50, "alpha_d": 0.50, "alpha_c": 0.40, "alpha_e": 0.15}),
    ("alpha_e=0.6",   {"tau_a": 0.50, "alpha_d": 0.50, "alpha_c": 0.40, "alpha_e": 0.60}),
]


def run_ofat_sweep(
    seeds: List[int] | None = None,
    total_train_steps: int = 1_300_000,
    rollout_steps: int = 4096,
    eval_episodes: int = 100,
    max_ep_steps: int = 1500,
    output_dir: str | Path = "outputs/s48_ofat",
    selected_labels: List[str] | None = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute Supplementary Table S48 OFAT training & evaluation sweep in the LF setting."""
    seed_list = [0, 1, 2, 3, 4] if seeds is None else list(seeds)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    for label, params in OFAT_PARAMETER_GRID:
        if selected_labels is not None and label not in selected_labels:
            continue

        full_rates: List[float] = []
        cf_rates: List[float] = []
        for s in seed_list:
            env_cfg = EnvConfig(
                max_steps=max_ep_steps,
                wireless=WirelessConfig(wireless_mode="lf"),
            )
            m_cfg = ModelConfig(
                method="mtc_mappo",
                tau_a=params["tau_a"],
                alpha_d=params["alpha_d"],
                alpha_c=params["alpha_c"],
                alpha_e=params["alpha_e"],
            )
            p_cfg = PPOConfig(
                total_env_steps=total_train_steps,
                rollout_steps=min(rollout_steps, total_train_steps),
                minibatch_size=min(512, total_train_steps * env_cfg.num_uavs),
            )
            trainer = CTDEMAPPOTrainer(env_cfg, m_cfg, p_cfg, seed=s, device=device)
            trainer.train()

            safe_lbl = label.replace("=", "_").replace(".", "p")
            save_checkpoint(
                out_root / f"ckpt_{safe_lbl}_seed{s}.pt",
                trainer.agent,
                env_cfg=env_cfg,
                ppo_cfg=p_cfg,
                seed=s,
            )

            eval_env = LowFidelityUAVEnv(cfg=env_cfg, training=False)
            ep_full = []
            ep_cf = []
            for ep in range(eval_episodes):
                obs, _ = eval_env.reset(seed=500_000 + 1000 * s + ep)
                done = False
                info: Dict[str, Any] = {}
                while not done:
                    act_out = trainer.agent.act(obs, deterministic=True, device=device)
                    obs, _, _, done, info = eval_env.step(
                        act_out["motion_cmds"], act_out["slot_powers"]
                    )
                m = info["episode_metrics"]
                ep_full.append(m["full_success"])
                ep_cf.append(m["cf_full_success"])

            full_rates.append(float(np.mean(ep_full)))
            cf_rates.append(float(np.mean(ep_cf)))

        ddof = 1 if len(seed_list) > 1 else 0
        results[label] = {
            "params": params,
            "n_seeds": len(seed_list),
            "full_mean": float(np.mean(full_rates)),
            "full_sd": float(np.std(full_rates, ddof=ddof)),
            "cf_full_mean": float(np.mean(cf_rates)),
            "cf_full_sd": float(np.std(cf_rates, ddof=ddof)),
            "seed_full_rates": full_rates,
            "seed_cf_rates": cf_rates,
        }

    (out_root / "s48_ofat_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Table S48 OFAT matching-score parameter sweep.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--total-steps", type=int, default=1_300_000)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--max-ep-steps", type=int, default=1500)
    parser.add_argument("--labels", type=str, nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/s48_ofat")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    res = run_ofat_sweep(
        seeds=args.seeds,
        total_train_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        eval_episodes=args.eval_episodes,
        max_ep_steps=args.max_ep_steps,
        output_dir=args.output_dir,
        selected_labels=args.labels,
        device=args.device,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
