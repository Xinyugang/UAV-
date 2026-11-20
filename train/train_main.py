from __future__ import annotations

import argparse
import json
from pathlib import Path

from configs.defaults import EnvConfig, ModelConfig, PPOConfig, WirelessConfig
from train.mappo_trainer import CTDEMAPPOTrainer, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CTDE-MAPPO training entrypoint for MTC-MAPPO, baselines, and ablations."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mtc_mappo",
        choices=[
            "mtc_mappo",
            "morph_gat",
            "gat_mappo",
            "mappo_morph",
            "role_mappo",
            "mappo",
            "tarmac_b1",
        ],
    )
    parser.add_argument("--wireless-mode", type=str, default="e1", choices=["lf", "e1"])
    parser.add_argument("--service-rule", type=str, default="ADD", choices=["ADD", "GEO", "HARM"])
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--total-steps", type=int, default=1_300_000)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--max-ep-steps", type=int, default=1500)
    parser.add_argument("--w4-comm", type=float, default=0.2, help="Comm reward weight w_4 (Table S8)")
    parser.add_argument("--homogeneous", action="store_true", help="E-HET homogeneous 5-UAV team")
    # Ablations (Table 5 / E2 & Table 7)
    parser.add_argument("--ablate-cap", action="store_true")
    parser.add_argument("--ablate-attn", action="store_true")
    parser.add_argument("--ablate-mod", action="store_true")
    parser.add_argument("--ablate-prior", action="store_true")
    # Matching-score parameters (Table S48)
    parser.add_argument("--tau-a", type=float, default=0.5)
    parser.add_argument("--alpha-d", type=float, default=0.5)
    parser.add_argument("--alpha-c", type=float, default=0.4)
    parser.add_argument("--alpha-e", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-path", type=str, default="checkpoints/mtc_mappo_seed1001.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    w_cfg = WirelessConfig(wireless_mode=args.wireless_mode)
    env_cfg = EnvConfig(
        max_steps=args.max_ep_steps,
        service_rule=args.service_rule,
        w4_comm=args.w4_comm,
        homogeneous_team=args.homogeneous,
        wireless=w_cfg,
    )
    model_cfg = ModelConfig(
        method=args.method,
        ablate_capability_descriptor=args.ablate_cap,
        ablate_task_attention=args.ablate_attn,
        ablate_link_energy_mod=args.ablate_mod,
        ablate_comm_prior=args.ablate_prior,
        tau_a=args.tau_a,
        alpha_d=args.alpha_d,
        alpha_c=args.alpha_c,
        alpha_e=args.alpha_e,
    )
    ppo_cfg = PPOConfig(
        total_env_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
    )

    trainer = CTDEMAPPOTrainer(
        env_cfg=env_cfg,
        model_cfg=model_cfg,
        ppo_cfg=ppo_cfg,
        seed=args.seed,
        device=args.device,
    )
    train_summary = trainer.train()
    ckpt_path = save_checkpoint(
        args.save_path,
        trainer.agent,
        env_cfg=env_cfg,
        ppo_cfg=ppo_cfg,
        seed=args.seed,
        extra_meta={"train_summary": train_summary},
    )
    out_json = Path(args.save_path).with_suffix(".json")
    out_json.write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "param_breakdown": trainer.agent.parameter_breakdown(),
                "total_env_steps": train_summary["total_env_steps"],
                "episodes_completed": train_summary["episodes_completed"],
                "recent_full_success": train_summary["recent_full_success"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved trained checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
