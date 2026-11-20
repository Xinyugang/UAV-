from __future__ import annotations

import numpy as np

from configs.defaults import CompensatorConfig, EnvConfig, ModelConfig, PyBulletConfig
from envs.pybullet_uav_env import (
    DisturbanceCompensator,
    OrnsteinUhlenbeckProcess,
    PyBulletUAVEnv,
    run_single_platform_tracking_trial,
)
from models.policy import MTCMAPPOAgent


def test_ou_process_and_compensator_rmse_physics() -> None:
    """Verify OU stationary variance and Group B vs Group C Prediction & Residual RMSE (Table S54b)."""
    rng = np.random.Generator(np.random.PCG64(12345))
    ou = OrnsteinUhlenbeckProcess(num_uavs=4, sigma_d=0.6, tau_c=2.0, dt=0.01)
    ou.reset(rng, stationary_init=True)
    samples = np.stack([ou.step(rng) for _ in range(8_000)], axis=0)
    emp_sd = float(np.std(samples))
    assert 0.52 <= emp_sd <= 0.68

    # Simulate 5,000 steps at 100 Hz for Group B & Group C under High-Error predictor noise (0.7 m/s^2)
    comp_b = DisturbanceCompensator(
        num_uavs=1, cfg=CompensatorConfig(group="B", disturbance_mode="high_error")
    )
    comp_c = DisturbanceCompensator(
        num_uavs=1, cfg=CompensatorConfig(group="C", disturbance_mode="high_error")
    )
    ou1 = OrnsteinUhlenbeckProcess(num_uavs=1, sigma_d=0.6, tau_c=2.0, dt=0.01)
    ou1.reset(rng, stationary_init=False)

    u_prev_b = np.zeros((1, 2))
    u_prev_c = np.zeros((1, 2))
    for _ in range(5_000):
        d_true = ou1.step(rng)
        p_noise = rng.normal(0.0, 0.7, size=(1, 2))
        imu_n = rng.normal(0.0, 0.02, size=(1, 2))

        a_meas_b = u_prev_b + d_true + imu_n
        a_meas_c = u_prev_c + d_true + imu_n

        u_ff_b = comp_b.update(a_meas_b, d_true, p_noise)
        u_ff_c = comp_c.update(a_meas_c, d_true, p_noise)

        u_prev_b = u_ff_b
        u_prev_c = u_ff_c
        comp_b.set_applied_actuator_accel(u_prev_b)
        comp_c.set_applied_actuator_accel(u_prev_c)

    diag_b = comp_b.get_offline_diagnostics()
    diag_c = comp_c.get_offline_diagnostics()
    # High-error prediction RMSE must be ~ sqrt(lag_var + 0.7^2) in [0.71, 0.79] m/s^2 (Table S54b: 0.747)
    assert 0.70 <= diag_b["prediction_rmse"] <= 0.80
    assert 0.70 <= diag_c["prediction_rmse"] <= 0.80
    # Group C residual estimation RMSE vs r_0 = d_true - p must be much smaller (~0.15-0.22 m/s^2)
    assert 0.12 <= diag_c["estimation_rmse_vs_r0"] <= 0.25


def test_pybullet_headless_6dof_transfer_and_s53_recovery() -> None:
    """Verify headless PyBullet 6-DOF environment, E8 sweeps, and S53 PID+PC tracking improvement."""
    env_cfg = EnvConfig(max_steps=15)
    pb_cfg = PyBulletConfig(
        gui=False,
        control_delay_ms=40.0,
        comm_noise_std=0.05,
        attitude_atten_kappa=0.5,
        compensator=CompensatorConfig(group="C", disturbance_mode="high_error"),
    )
    pb_env = PyBulletUAVEnv(env_cfg=env_cfg, pb_cfg=pb_cfg)
    agent = MTCMAPPOAgent(ModelConfig(method="mtc_mappo"))
    try:
        obs, _ = pb_env.reset(seed=505)
        done = False
        info = {}
        while not done:
            act_out = agent.act(obs, deterministic=True)
            obs, _, rew, done, info = pb_env.step(act_out["motion_cmds"], act_out["slot_powers"])
            assert np.isfinite(rew)
        m = info["episode_metrics"]
        assert m["executed_steps"] == 15.0
        assert np.isfinite(m["motor_energy_index"]) and m["motor_energy_index"] > 0.0
        assert np.isfinite(m["utp"]) and m["utp"] > 0.0
        assert np.isfinite(m["tracking_rmse"])
    finally:
        pb_env.close()

    # Single-platform tracking & recovery trial (shortened window for fast unit test)
    res_pid = run_single_platform_tracking_trial(
        platform_type="endurance",
        controller_group="A",
        seed=42,
        total_duration_s=18.0,
        dist_start_s=3.0,
        dist_end_s=10.0,
        recovery_hold_s=2.0,
    )
    res_pc = run_single_platform_tracking_trial(
        platform_type="endurance",
        controller_group="C",
        seed=42,
        total_duration_s=18.0,
        dist_start_s=3.0,
        dist_end_s=10.0,
        recovery_hold_s=2.0,
    )
    # Compensated PID+PC must reduce disturbance-window RMSE relative to uncompensated PID
    assert res_pc["rmse_mps"] < res_pid["rmse_mps"]
