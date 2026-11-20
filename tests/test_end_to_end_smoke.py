from __future__ import annotations

from pathlib import Path
import numpy as np

from configs.defaults import EnvConfig, ModelConfig, PPOConfig, WirelessConfig
from eval.eval_benchmarks import evaluate_policy_episodes, evaluate_s6_allocation_diagnostics
from eval.eval_interference import (
    evaluate_12frame_radio_window_replay,
    evaluate_prior_mixing_intervention,
    evaluate_s30_open_loop_wireless,
    evaluate_s33_asymmetric_exposure,
    evaluate_s36_exposure_rotation,
)
from eval.eval_masking import evaluate_distance_matched_masking, evaluate_lf_attention_masking
from eval.eval_pybullet import evaluate_s54_closed_loop_compensation
from eval.eval_scaling_timing import evaluate_team_size_scaling
from eval.eval_spearman import evaluate_s44_candidate_correlations
from train.mappo_trainer import CTDEMAPPOTrainer, load_checkpoint, save_checkpoint
from train.sweep_ofat import run_ofat_sweep


def test_end_to_end_training_and_all_eval_suites_smoke(tmp_path: Path) -> None:
    """Run complete end-to-end smoke verification of CTDE-MAPPO training and all evaluation suites."""
    env_cfg = EnvConfig(max_steps=16, wireless=WirelessConfig(wireless_mode="e1"))
    mtc_cfg = ModelConfig(method="mtc_mappo")
    gat_cfg = ModelConfig(method="gat_mappo")
    ppo_cfg = PPOConfig(total_env_steps=32, rollout_steps=16, minibatch_size=32, ppo_epochs=2)

    # 1. Train MTC-MAPPO and GAT-MAPPO for 2 rollouts & verify checkpoint round-trip
    trainer_mtc = CTDEMAPPOTrainer(env_cfg, mtc_cfg, ppo_cfg, seed=1001)
    summary_mtc = trainer_mtc.train()
    assert summary_mtc["total_env_steps"] == 32
    ckpt_path = save_checkpoint(
        tmp_path / "mtc_smoke.pt", trainer_mtc.agent, env_cfg, ppo_cfg, seed=1001
    )
    loaded_mtc, _ = load_checkpoint(ckpt_path)

    trainer_gat = CTDEMAPPOTrainer(env_cfg, gat_cfg, ppo_cfg, seed=1001)
    trainer_gat.train()
    gat_agent = trainer_gat.agent

    # 2. Closed-loop benchmark + S6 allocation entropy/complement identity (C = 1 - H)
    bench = evaluate_policy_episodes(loaded_mtc, env_cfg, num_episodes=3)
    assert bench["num_episodes"] == 3
    s6 = evaluate_s6_allocation_diagnostics(loaded_mtc, env_cfg, num_episodes=3)
    if s6["allocation_entropy"] > 0.0:
        assert abs((s6["allocation_entropy"] + s6["entropy_complement"]) - 1.0) < 1e-9

    # 3. OFAT score sensitivity sweep smoke (S48)
    ofat_res = run_ofat_sweep(
        seeds=[0],
        total_train_steps=16,
        rollout_steps=16,
        eval_episodes=2,
        max_ep_steps=12,
        output_dir=tmp_path / "ofat",
        selected_labels=["default", "tau_a=0.25"],
    )
    assert "default" in ofat_res and "tau_a=0.25" in ofat_res

    # 4. LF & Distance-matched attention masking (Fig. 5, S45-S46)
    lf_mask = evaluate_lf_attention_masking(
        loaded_mtc, env_cfg, num_episodes=2, conditions=("no_mask", "mask_top", "randomize_attention")
    )
    assert set(lf_mask.keys()) == {"no_mask", "mask_top", "randomize_attention"}

    dm_mask = evaluate_distance_matched_masking([loaded_mtc], env_cfg, episodes_per_ckpt=4)
    assert "top_minus_distance_matched" in dm_mask["contrasts_pp"]

    # 5. Prior mixing (S40-S43), 12-frame radio-window replay (S42), & open-loop S30/S33/S36
    mix_res = evaluate_prior_mixing_intervention(
        [loaded_mtc], [gat_agent], env_cfg, episodes_per_ckpt=2, power_levels_w=(0.0, 2.0)
    )
    assert "2W" in mix_res["summary_s41"]

    s42_res = evaluate_12frame_radio_window_replay(
        [loaded_mtc],
        [gat_agent],
        env_cfg,
        windows_per_model=2,
        window_start_frame=2,
        window_len=12,
        power_levels_w=(0.0, 2.0),
    )
    # Fixed-allocation replay invariant: entropy must be identical across 0 W and 2 W!
    assert abs(s42_res["0W_MTC"]["entropy"]["mean"] - s42_res["2W_MTC"]["entropy"]["mean"]) < 1e-12

    s30 = evaluate_s30_open_loop_wireless()
    s33 = evaluate_s33_asymmetric_exposure()
    s36 = evaluate_s36_exposure_rotation(n_boot=200)
    assert "SI-QK" in s30 and "SI-Mix" in s30
    assert s33["0W_Task-prior"]["utility"] > s33["2W_Task-prior"]["utility"]
    # Exposure rotation reverses 2-W capability-minus-balanced utility (DiD < 0, Table S36)
    assert s36["cap_minus_bal_2W_aligned"] < 0.0 < s36["cap_minus_bal_2W_rotated"]
    assert s36["did_aligned_minus_rotated"] < -0.1

    # 6. Candidate-level Spearman + 7-point support + 50-step masked return loss (S44, S49, S50)
    s44 = evaluate_s44_candidate_correlations(
        [loaded_mtc],
        env_cfg,
        states_per_model=10,
        behavior_states_per_model=3,
        behavior_horizon=8,
    )
    assert "Behavior" in s44["table_s44"]
    pct_sum_d0 = sum(s44["table_s50"]["D0"].values())
    assert abs(pct_sum_d0 - 100.0) < 1e-6 or pct_sum_d0 == 0.0

    # 7. Team-size scaling N=5/10/20 & single-thread CPU latency (S51-S52)
    s51 = evaluate_team_size_scaling(
        [loaded_mtc],
        [gat_agent],
        env_cfg,
        team_sizes=(5, 10, 20),
        episodes_per_ckpt=2,
        warmup_passes=5,
        timed_passes=20,
    )
    assert "N=20" in s51["table_s51"]
    assert s51["table_s51"]["N=20"]["MTC"]["time_ms"]["mean"] > 0.0

    # 8. PyBullet 5-UAV frozen-policy Group A/B/C closed-loop compensation smoke (S54-S56)
    short_pb_env_cfg = EnvConfig(max_steps=8, wireless=WirelessConfig(wireless_mode="e1"))
    s54 = evaluate_s54_closed_loop_compensation(
        [loaded_mtc], short_pb_env_cfg, episodes_per_cell=2
    )
    assert "high_error_C_minus_B" in s54["table_s55_table12"]
    # Verify Group A shared disturbed run set reuse: low_error_A == high_error_A identically
    assert (
        s54["table_s54"]["low_error_A"]["tracking_rmse"]["mean"]
        == s54["table_s54"]["high_error_A"]["tracking_rmse"]["mean"]
    )
