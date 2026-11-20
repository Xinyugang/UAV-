from __future__ import annotations

from collections import deque
import copy
import time
from typing import Any, Deque, Dict, List, Tuple
import numpy as np
import pybullet as p

from configs.defaults import (
    CompensatorConfig,
    EnvConfig,
    PLATFORM_SPECS,
    PyBulletConfig,
)
from envs.lf_uav_env import LowFidelityUAVEnv
from envs.service_and_reward import (
    compute_routing_weights,
    compute_s13_contribution_alignment,
    compute_service_quality,
    compute_specialization_index,
    compute_step_energy,
    compute_step_reward,
)


class OrnsteinUhlenbeckProcess:
    """Two-axis Ornstein-Uhlenbeck equivalent acceleration disturbance (Tables S54-S56).

    Continuous SDE: dd = -(1/tau_c) d dt + q dW,  q = sigma_d * sqrt(2 / tau_c).
    Exact discrete transition at step dt:
        d_{k+1} = alpha * d_k + sigma_d * sqrt(1 - alpha^2) * N(0, I_2),  alpha = exp(-dt / tau_c).
    Stationary per-axis SD equals exact sigma_d (0.6 m/s^2).
    """

    def __init__(self, num_uavs: int, sigma_d: float = 0.6, tau_c: float = 2.0, dt: float = 0.01) -> None:
        self.num_uavs = num_uavs
        self.sigma_d = float(sigma_d)
        self.tau_c = float(tau_c)
        self.dt = float(dt)
        self.alpha = float(np.exp(-self.dt / max(self.tau_c, 1e-6)))
        self.step_std = self.sigma_d * float(np.sqrt(max(0.0, 1.0 - self.alpha ** 2)))
        self.state = np.zeros((num_uavs, 2), dtype=np.float64)

    def reset(self, rng: np.random.Generator, stationary_init: bool = False) -> None:
        if stationary_init:
            self.state = rng.normal(0.0, self.sigma_d, size=(self.num_uavs, 2))
        else:
            self.state = np.zeros((self.num_uavs, 2), dtype=np.float64)

    def step(self, rng: np.random.Generator) -> np.ndarray:
        noise = rng.normal(0.0, 1.0, size=(self.num_uavs, 2))
        self.state = self.alpha * self.state + self.step_std * noise
        return self.state.copy()


class DisturbanceCompensator:
    """100-Hz Disturbance-Prediction Feedforward & First-Order Residual Estimator (S53-S56).

    Implements exact update equations in Supplementary Material Section 'Prediction and
    residual compensation interface':
        d_tilde_k = a_meas,k - u_hat_{k-1}
        p_k       = p_{k-1} + (dt / tau_f) * (d_tilde_{k-1} - p_{k-1})
        p'_k      = p_k + n_k
        r_hat_k   = r_hat_{k-1} + K_e * (d_tilde_{k-1} - p'_{k-1} - r_hat_{k-1})
        u_ff^A    = 0
        u_ff^B    = -clip(p'_k, -a_max, a_max)
        u_ff^C    = -clip(p'_k + r_hat_k, -a_max, a_max)
    """

    def __init__(self, num_uavs: int, cfg: CompensatorConfig) -> None:
        self.num_uavs = num_uavs
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        n = self.num_uavs
        self.d_tilde_prev = np.zeros((n, 2), dtype=np.float64)
        self.p_state = np.zeros((n, 2), dtype=np.float64)
        self.p_prime_prev = np.zeros((n, 2), dtype=np.float64)
        self.r_hat = np.zeros((n, 2), dtype=np.float64)
        self.u_hat_prev = np.zeros((n, 2), dtype=np.float64)
        self.has_prev = False

        # Offline error accumulators (two-axis component MSE: mean((e_x^2 + e_y^2)/2))
        self.pred_sq_err_sum: float = 0.0
        self.res_sq_err_sum: float = 0.0
        self.sample_count: int = 0
        self.total_comp_time_s: float = 0.0
        self.comp_calls: int = 0

    def update(
        self,
        a_meas: np.ndarray,
        d_true_offline: np.ndarray,
        pred_noise: np.ndarray,
    ) -> np.ndarray:
        """Execute one 100-Hz estimator step for all UAVs and return u_ff of shape (N, 2)."""
        grp = self.cfg.group.upper()
        if grp == "A":
            return np.zeros_like(a_meas)

        t0 = time.perf_counter()
        d_tilde_k = a_meas - self.u_hat_prev

        if self.has_prev:
            alpha_f = self.cfg.dt_ctrl / max(self.cfg.tau_f, 1e-6)
            p_k = self.p_state + alpha_f * (self.d_tilde_prev - self.p_state)
            if grp == "C":
                r_hat_k = self.r_hat + self.cfg.k_e * (
                    self.d_tilde_prev - self.p_prime_prev - self.r_hat
                )
            else:
                r_hat_k = np.zeros_like(self.r_hat)
        else:
            p_k = np.zeros_like(self.p_state)
            r_hat_k = np.zeros_like(self.r_hat)

        p_prime_k = p_k + pred_noise
        a_lim = self.cfg.a_max_ff
        if grp == "B":
            u_ff = -np.clip(p_prime_k, -a_lim, a_lim)
        elif grp == "C":
            u_ff = -np.clip(p_prime_k + r_hat_k, -a_lim, a_lim)
        else:
            raise ValueError(f"Unsupported compensator group: {self.cfg.group}")

        self.d_tilde_prev = d_tilde_k
        self.p_state = p_k
        self.p_prime_prev = p_prime_k
        self.r_hat = r_hat_k
        self.has_prev = True
        self.total_comp_time_s += time.perf_counter() - t0
        self.comp_calls += a_meas.shape[0]

        # Offline RMSE logging (ground truth d_true_offline never enters online control)
        err_pred = d_true_offline - p_prime_k
        r0_ref = d_true_offline - p_k
        err_res = r0_ref - r_hat_k
        self.pred_sq_err_sum += float(np.sum(0.5 * np.sum(err_pred ** 2, axis=-1)))
        self.res_sq_err_sum += float(np.sum(0.5 * np.sum(err_res ** 2, axis=-1)))
        self.sample_count += a_meas.shape[0]
        return u_ff

    def set_applied_actuator_accel(self, u_hat_applied: np.ndarray) -> None:
        self.u_hat_prev = u_hat_applied.copy()

    def get_offline_diagnostics(self) -> Dict[str, float]:
        denom = max(1, self.sample_count)
        pred_rmse = float(np.sqrt(self.pred_sq_err_sum / denom))
        est_rmse = float(np.sqrt(self.res_sq_err_sum / denom))
        extra_ms = (
            1000.0 * self.total_comp_time_s / max(1, self.comp_calls)
            if self.cfg.group.upper() in ("B", "C")
            else 0.0
        )
        return {
            "prediction_rmse": pred_rmse,
            "estimation_rmse_vs_r0": est_rmse,
            "extra_comp_time_ms": extra_ms,
        }


class PyBulletUAVEnv(LowFidelityUAVEnv):
    """Six-Degree-of-Freedom PyBullet Quadrotor Environment with 100 Hz Cascaded PID.

    Inherits high-level mission, task, wireless SINR, and observation interfaces from
    LowFidelityUAVEnv so frozen E1 / LF checkpoints transfer zero-shot with identical
    observation dimensions, while replacing 2D point kinematics with:
      - 6-DOF rigid-body dynamics in PyBullet (DIRECT headless or GUI)
      - Mass-scaled inertia, gravity, attitude limits, thrust saturation
      - 30 ms first-order motor dynamics & 100 Hz cascaded velocity-attitude PID
      - E8 control delay, communication noise, and attitude attenuation (G_ctrl)
      - Group A / B / C disturbance-prediction feedforward & residual compensation (S54-S56)
    """

    def __init__(
        self,
        env_cfg: EnvConfig | None = None,
        pb_cfg: PyBulletConfig | None = None,
    ) -> None:
        super().__init__(cfg=env_cfg, training=False)
        self.pb_cfg = copy.deepcopy(pb_cfg) if pb_cfg is not None else PyBulletConfig()
        self.dt_ctrl: float = 1.0 / float(self.pb_cfg.ctrl_freq_hz)
        self.substeps_per_policy: int = max(
            1, int(round(self.cfg.dt / self.dt_ctrl))
        )  # 0.2 / 0.01 = 20

        mode = p.GUI if self.pb_cfg.gui else p.DIRECT
        self.client_id: int = p.connect(mode)
        if self.client_id < 0:
            raise RuntimeError("Failed to initialize PyBullet physics client.")

        self.uav_body_ids: List[int] = []
        self.obs_body_ids: List[int] = []

        # Controller internal states
        n = self.cfg.num_uavs
        self.vel_int_err = np.zeros((n, 2), dtype=np.float64)
        self.prev_vel_3d = np.zeros((n, 3), dtype=np.float64)
        self.motor_thrust_act = np.zeros(n, dtype=np.float64)
        self.motor_torque_act = np.zeros((n, 3), dtype=np.float64)
        self.cmd_delay_queues: List[Deque[np.ndarray]] = [deque() for _ in range(n)]

        self.ou_process = OrnsteinUhlenbeckProcess(
            num_uavs=n,
            sigma_d=self.pb_cfg.compensator.ou_sigma_d,
            tau_c=self.pb_cfg.compensator.ou_tau_c,
            dt=self.dt_ctrl,
        )
        self.compensator = DisturbanceCompensator(n, self.pb_cfg.compensator)
        self.dist_rng = np.random.Generator(np.random.PCG64(1))
        self.pred_noise_rng = np.random.Generator(np.random.PCG64(2))

        # PyBullet control & motor-energy accumulators
        self.motor_energy_index_accum: float = 0.0
        self.tracking_sq_err_sum: float = 0.0
        self.saturated_substeps: int = 0
        self.total_ctrl_substeps: int = 0
        self.att_sq_sum: float = 0.0

    def close(self) -> None:
        if getattr(self, "client_id", -1) >= 0:
            try:
                p.disconnect(physicsClientId=self.client_id)
            except Exception:
                pass
            self.client_id = -1

    def __del__(self) -> None:
        self.close()

    def _rebuild_pybullet_world(self) -> None:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -self.pb_cfg.gravity, physicsClientId=self.client_id)
        p.setTimeStep(self.dt_ctrl, physicsClientId=self.client_id)

        # Ground plane
        ground_col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client_id)
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=ground_col,
            basePosition=[0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )

        # Cylindrical obstacles
        self.obs_body_ids = []
        if self.obstacle_centers.shape[0] > 0:
            obs_col = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=self.cfg.d_col_obs,
                height=20.0,
                physicsClientId=self.client_id,
            )
            for oc in self.obstacle_centers:
                bid = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=obs_col,
                    basePosition=[float(oc[0]), float(oc[1]), 10.0],
                    physicsClientId=self.client_id,
                )
                self.obs_body_ids.append(bid)

        # 6-DOF Quadrotor bodies with platform-specific mass & mass-scaled inertia
        self.uav_body_ids = []
        uav_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.18, 0.18, 0.06], physicsClientId=self.client_id
        )
        for i in range(self.cfg.num_uavs):
            mass_i = float(self.uav_caps[i, 0])
            bid = p.createMultiBody(
                baseMass=mass_i,
                baseCollisionShapeIndex=uav_col,
                basePosition=[
                    float(self.uav_pos[i, 0]),
                    float(self.uav_pos[i, 1]),
                    self.pb_cfg.nominal_altitude_m,
                ],
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client_id,
            )
            # Mass-scaled quadrotor inertia
            p.changeDynamics(
                bid,
                -1,
                localInertiaDiagonal=[0.012 * mass_i, 0.012 * mass_i, 0.022 * mass_i],
                linearDamping=0.0,
                angularDamping=0.04,
                physicsClientId=self.client_id,
            )
            self.uav_body_ids.append(bid)

    def reset(
        self,
        seed: int | None = None,
        demand_seed: int | None = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        base_seed = 0 if seed is None else int(seed)
        # Separate deterministic streams so Group A/B/C share identical true OU disturbances
        self.dist_rng = np.random.Generator(np.random.PCG64(base_seed + 700_000))
        self.pred_noise_rng = np.random.Generator(np.random.PCG64(base_seed + 900_000))

        obs_tuple = super().reset(seed=seed, demand_seed=demand_seed)
        self._rebuild_pybullet_world()

        n = self.cfg.num_uavs
        self.vel_int_err = np.zeros((n, 2), dtype=np.float64)
        self.prev_vel_3d = np.zeros((n, 3), dtype=np.float64)
        self.motor_thrust_act = self.uav_caps[:, 0] * self.pb_cfg.gravity
        self.motor_torque_act = np.zeros((n, 3), dtype=np.float64)

        delay_steps = max(0, int(round(self.pb_cfg.control_delay_ms / (1000.0 * self.dt_ctrl))))
        self.cmd_delay_queues = []
        for _ in range(n):
            dq: Deque[np.ndarray] = deque()
            for _s in range(delay_steps + 1):
                dq.append(np.zeros(2, dtype=np.float64))
            self.cmd_delay_queues.append(dq)

        self.ou_process.num_uavs = n
        self.ou_process.reset(self.dist_rng, stationary_init=False)
        self.compensator = DisturbanceCompensator(n, self.pb_cfg.compensator)

        self.motor_energy_index_accum = 0.0
        self.tracking_sq_err_sum = 0.0
        self.saturated_substeps = 0
        self.total_ctrl_substeps = 0
        self.att_sq_sum = 0.0

        if self.pb_cfg.comm_noise_std > 0.0:
            return self._build_observations(comm_noise_std=self.pb_cfg.comm_noise_std)
        return obs_tuple

    def _step_100hz_substep(
        self,
        v_ref_all: np.ndarray,
        external_accel_dist: np.ndarray,
        pred_noise: np.ndarray,
    ) -> None:
        """Advance all 6-DOF quadrotors by one 100-Hz (0.01 s) cascaded PID + motor step."""
        n = self.cfg.num_uavs
        g = self.pb_cfg.gravity
        g_ctrl = 1.0 / (1.0 + max(0.0, self.pb_cfg.attitude_atten_kappa))
        alpha_motor = self.dt_ctrl / (self.pb_cfg.tau_motor + self.dt_ctrl)

        pos_3d = np.zeros((n, 3), dtype=np.float64)
        vel_3d = np.zeros((n, 3), dtype=np.float64)
        eulers = np.zeros((n, 3), dtype=np.float64)
        ang_vel_b = np.zeros((n, 3), dtype=np.float64)
        rot_mats = np.zeros((n, 3, 3), dtype=np.float64)

        for i, bid in enumerate(self.uav_body_ids):
            pos_w, orn_w = p.getBasePositionAndOrientation(bid, physicsClientId=self.client_id)
            lin_w, ang_w = p.getBaseVelocity(bid, physicsClientId=self.client_id)
            pos_3d[i] = pos_w
            vel_3d[i] = lin_w
            eulers[i] = p.getEulerFromQuaternion(orn_w)
            rot_mat = np.asarray(p.getMatrixFromQuaternion(orn_w), dtype=np.float64).reshape(3, 3)
            rot_mats[i] = rot_mat
            ang_vel_b[i] = rot_mat.T @ np.asarray(ang_w, dtype=np.float64)

        # Simulated onboard IMU horizontal acceleration (specific force with gravity removed)
        a_kin = (vel_3d[:, :2] - self.prev_vel_3d[:, :2]) / self.dt_ctrl
        imu_noise = self.dist_rng.normal(0.0, 0.015, size=(n, 2))
        a_meas = a_kin + imu_noise
        self.prev_vel_3d = vel_3d.copy()

        # Disturbance prediction feedforward + residual compensation (Groups A / B / C)
        u_ff = self.compensator.update(
            a_meas=a_meas,
            d_true_offline=external_accel_dist,
            pred_noise=pred_noise,
        )

        u_hat_applied = np.zeros((n, 2), dtype=np.float64)
        for i, bid in enumerate(self.uav_body_ids):
            mass_i = float(self.uav_caps[i, 0])
            a_max_i = float(self.uav_caps[i, 3])
            depleted = self.uav_energy_j[i] <= 0.0

            v_ref_i = np.zeros(2, dtype=np.float64) if depleted else v_ref_all[i]
            e_v = v_ref_i - vel_3d[i, :2]
            self.tracking_sq_err_sum += float(np.sum(e_v ** 2))
            self.total_ctrl_substeps += 1

            # Anti-windup integral state
            self.vel_int_err[i] = np.clip(
                self.vel_int_err[i] + e_v * self.dt_ctrl, -0.85, 0.85
            )

            # Outer horizontal velocity PID + execution-layer feedforward u_ff[i]
            a_pid = g_ctrl * (
                self.pb_cfg.kp_pos * e_v
                - self.pb_cfg.kd_pos * (0.15 * vel_3d[i, :2])
                + self.pb_cfg.ki_pos * self.vel_int_err[i]
            )
            a_des = a_pid + u_ff[i]

            # Enforce horizontal acceleration / tilt limits & check actuator saturation
            a_lim_horiz = min(g * np.tan(self.pb_cfg.max_tilt_rad), 1.6 * a_max_i)
            a_norm = float(np.linalg.norm(a_des))
            sat_flag = False
            if a_norm > a_lim_horiz:
                a_des = a_des * (a_lim_horiz / max(a_norm, 1e-12))
                sat_flag = True

            # Desired roll & pitch in right-handed ENU frame:
            #   +pitch (around +Y) -> +X accel;  -roll (around +X) -> +Y accel
            roll_des = float(
                np.clip(-a_des[1] / g, -self.pb_cfg.max_tilt_rad, self.pb_cfg.max_tilt_rad)
            )
            pitch_des = float(
                np.clip(a_des[0] / g, -self.pb_cfg.max_tilt_rad, self.pb_cfg.max_tilt_rad)
            )
            yaw_des = 0.0

            # Altitude hold around nominal_altitude_m
            e_z = self.pb_cfg.nominal_altitude_m - pos_3d[i, 2]
            a_z = 4.0 * e_z - 2.8 * vel_3d[i, 2]
            cos_tilt = max(0.5, float(np.cos(eulers[i, 0]) * np.cos(eulers[i, 1])))
            thrust_raw = mass_i * max(0.0, g + a_z) / cos_tilt
            thrust_max = self.pb_cfg.thrust_margin_ratio * mass_i * g
            if thrust_raw > thrust_max or thrust_raw <= 0.0:
                sat_flag = True
            thrust_cmd = 0.0 if depleted else float(np.clip(thrust_raw, 0.0, thrust_max))
            if sat_flag:
                self.saturated_substeps += 1

            # Inner attitude PD torque loop (scaled to rigid-body inertia ~0.012*m)
            att_err = np.array(
                [roll_des - eulers[i, 0], pitch_des - eulers[i, 1], yaw_des - eulers[i, 2]],
                dtype=np.float64,
            )
            # Wrap yaw error to [-pi, pi]
            att_err[2] = (att_err[2] + np.pi) % (2.0 * np.pi) - np.pi
            torque_scale = 2.2 * mass_i * g_ctrl
            torque_cmd = torque_scale * (
                self.pb_cfg.kp_att * att_err - self.pb_cfg.kd_att * ang_vel_b[i]
            )
            torque_cmd = np.clip(torque_cmd, -1.5 * mass_i, 1.5 * mass_i)

            # 30 ms first-order motor response
            self.motor_thrust_act[i] = (
                (1.0 - alpha_motor) * self.motor_thrust_act[i] + alpha_motor * thrust_cmd
            )
            self.motor_torque_act[i] = (
                (1.0 - alpha_motor) * self.motor_torque_act[i] + alpha_motor * torque_cmd
            )

            # Exact SE(3) world-frame horizontal acceleration u_hat from body +Z thrust
            u_hat_applied[i] = (self.motor_thrust_act[i] / mass_i) * rot_mats[i, :2, 2]
            self.att_sq_sum += float(eulers[i, 0] ** 2 + eulers[i, 1] ** 2)

            # Accumulate PyBullet motor-energy index (normalized so 5-UAV 300s nominal ~ 100)
            hover_w = mass_i * g
            effort = (self.motor_thrust_act[i] / max(hover_w, 1e-6)) ** 1.5 + 0.15 * float(
                np.linalg.norm(self.motor_torque_act[i]) / max(mass_i, 1e-6)
            )
            self.motor_energy_index_accum += 0.068 * effort * self.dt_ctrl

            # Apply body-frame rotor thrust & attitude torque
            p.applyExternalForce(
                bid,
                -1,
                forceObj=[0.0, 0.0, float(self.motor_thrust_act[i])],
                posObj=[0.0, 0.0, 0.0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client_id,
            )
            p.applyExternalTorque(
                bid,
                -1,
                torqueObj=self.motor_torque_act[i].tolist(),
                flags=p.LINK_FRAME,
                physicsClientId=self.client_id,
            )
            # Apply world-frame equivalent horizontal acceleration disturbance at body CoM pos_3d[i]
            if np.any(external_accel_dist[i] != 0.0):
                f_dist = [
                    float(mass_i * external_accel_dist[i, 0]),
                    float(mass_i * external_accel_dist[i, 1]),
                    0.0,
                ]
                p.applyExternalForce(
                    bid,
                    -1,
                    forceObj=f_dist,
                    posObj=pos_3d[i].tolist(),
                    flags=p.WORLD_FRAME,
                    physicsClientId=self.client_id,
                )

        self.compensator.set_applied_actuator_accel(u_hat_applied)
        p.stepSimulation(physicsClientId=self.client_id)

    def step(
        self,
        motion_cmds: np.ndarray,
        slot_powers: np.ndarray,
        ground_power_override: float | None = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute 20 PyBullet 100-Hz control steps (0.2 s) followed by mission/SINR update."""
        n = self.cfg.num_uavs
        u_cmd = np.asarray(motion_cmds, dtype=np.float64).copy()
        v_max = self.uav_caps[:, 2]
        cmd_norm = np.linalg.norm(u_cmd, axis=-1, keepdims=True)
        u_clipped = np.where(
            cmd_norm > v_max[:, None], u_cmd * (v_max[:, None] / np.maximum(cmd_norm, 1e-12)), u_cmd
        )

        delay_steps = max(0, int(round(self.pb_cfg.control_delay_ms / (1000.0 * self.dt_ctrl))))
        dist_mode = self.pb_cfg.compensator.disturbance_mode.lower()

        for _sub in range(self.substeps_per_policy):
            v_ref_sub = np.zeros((n, 2), dtype=np.float64)
            for i in range(n):
                self.cmd_delay_queues[i].append(u_clipped[i].copy())
                while len(self.cmd_delay_queues[i]) > delay_steps + 1:
                    self.cmd_delay_queues[i].popleft()
                v_ref_sub[i] = self.cmd_delay_queues[i][0]

            if dist_mode in ("low_error", "high_error"):
                d_true = self.ou_process.step(self.dist_rng)
            else:
                d_true = np.zeros((n, 2), dtype=np.float64)

            if dist_mode == "high_error":
                p_noise = self.pred_noise_rng.normal(
                    0.0, self.pb_cfg.compensator.high_error_noise_std, size=(n, 2)
                )
            else:
                p_noise = np.zeros((n, 2), dtype=np.float64)

            self._step_100hz_substep(v_ref_sub, d_true, p_noise)

        # Read back post-substep planar positions & velocities from PyBullet
        oob_cnt = 0
        for i, bid in enumerate(self.uav_body_ids):
            pos_w, orn_w = p.getBasePositionAndOrientation(bid, physicsClientId=self.client_id)
            lin_w, ang_w = p.getBaseVelocity(bid, physicsClientId=self.client_id)
            px, py, pz = float(pos_w[0]), float(pos_w[1]), float(pos_w[2])
            if px < 0.0 or px > self.cfg.workspace_size or py < 0.0 or py > self.cfg.workspace_size:
                oob_cnt += 1
                px_c = float(np.clip(px, 0.0, self.cfg.workspace_size))
                py_c = float(np.clip(py, 0.0, self.cfg.workspace_size))
                p.resetBasePositionAndOrientation(
                    bid, [px_c, py_c, max(0.5, pz)], orn_w, physicsClientId=self.client_id
                )
                p.resetBaseVelocity(bid, [0.0, 0.0, lin_w[2]], ang_w, physicsClientId=self.client_id)
                self.uav_pos[i] = [px_c, py_c]
                self.uav_vel[i] = [0.0, 0.0]
            else:
                self.uav_pos[i] = [px, py]
                self.uav_vel[i] = [float(lin_w[0]), float(lin_w[1])]
        self.total_oob += oob_cnt

        # Evaluate wireless SINR, task service, safety, and energy identically to LF interface
        p_mat = self.build_power_matrix_from_slot_powers(slot_powers)
        radio_out = self.wireless_channel.evaluate_step_links(
            self.uav_pos, p_mat, self.rng, ground_power_override=ground_power_override
        )
        attempted = radio_out["attempted"]
        effective = radio_out["effective"]
        sinr_lin = radio_out["sinr_linear"]
        sinr_db = radio_out["sinr_db"]

        n_att_step = int(np.sum(attempted))
        n_eff_step = int(np.sum(effective))
        self.attempted_links_total += n_att_step
        self.effective_links_total += n_eff_step
        if n_att_step > 0:
            self.attempted_sinr_db_sum += float(np.sum(sinr_db[attempted]))

        chi_comm = np.zeros(n, dtype=np.float64)
        for i, cand in enumerate(self.latest_candidates):
            m_i = len(cand.valid_ids)
            eff_i = sum(int(effective[i, uid]) for uid in cand.valid_ids)
            chi_comm[i] = eff_i / (m_i + self.cfg.wireless.eps_comm)

        _, alpha_env, _ = compute_routing_weights(
            self.uav_pos,
            self.task_centers,
            self.task_progress,
            self.task_req,
            gamma_d=self.cfg.gamma_d,
            gamma_u=self.cfg.gamma_u,
            d_norm=self.cfg.d_norm,
            routing_mode=self.cfg.routing_mode,
            r_alt=self.cfg.r_service,
        )
        q_mat = compute_service_quality(
            uav_positions=self.uav_pos,
            uav_caps=self.uav_caps,
            uav_energy_j=self.uav_energy_j,
            chi_comm=chi_comm,
            task_centers=self.task_centers,
            task_demands=self.task_demands,
            service_rule=self.cfg.service_rule,
            beta_d=self.cfg.beta_d,
            v_max_ref=self.cfg.v_max_ref,
            e_max_ref_j=self.cfg.e_max_ref_j,
        )
        routed_q = alpha_env * q_mat
        self.routed_service_accum += routed_q
        prev_prog = self.task_progress.copy()
        self.task_progress = self.task_progress + np.sum(routed_q, axis=0) * self.cfg.dt

        uav_uav_col = 0
        nearmiss_cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = float(np.linalg.norm(self.uav_pos[i] - self.uav_pos[j]))
                if d_ij < self.cfg.d_col_uav:
                    uav_uav_col += 1
                elif d_ij < self.cfg.d_col_uav + self.cfg.d_nearmiss_margin:
                    nearmiss_cnt += 1

        uav_obs_col = 0
        if self.obstacle_centers.shape[0] > 0:
            d_uo = np.linalg.norm(
                self.uav_pos[:, None, :] - self.obstacle_centers[None, :, :], axis=-1
            )
            col_mask_uo = d_uo < self.cfg.d_col_obs
            nm_mask_uo = (~col_mask_uo) & (
                d_uo < self.cfg.d_col_obs + self.cfg.d_nearmiss_margin
            )
            uav_obs_col = int(np.sum(col_mask_uo))
            nearmiss_cnt += int(np.sum(nm_mask_uo))

        step_collisions = uav_uav_col + uav_obs_col
        self.total_uav_uav_collisions += uav_uav_col
        self.total_uav_obs_collisions += uav_obs_col
        self.total_collisions += step_collisions
        self.total_nearmisses += nearmiss_cnt

        delta_e_j, power_terms = compute_step_energy(
            self.uav_caps, self.uav_vel, p_mat, self.uav_pos, self.task_centers, self.cfg
        )
        self.uav_energy_j = np.maximum(0.0, self.uav_energy_j - delta_e_j)
        self.total_energy_j += float(np.sum(delta_e_j))
        depleted_cnt = int(np.sum(self.uav_energy_j <= 0.0))
        self.total_energy_depleted += int(depleted_cnt > 0)

        reward, reward_terms = compute_step_reward(
            prev_progress=prev_prog,
            new_progress=self.task_progress,
            task_req=self.task_req,
            delta_e_j=delta_e_j,
            n_collisions=step_collisions,
            chi_comm=chi_comm,
            n_constraint_violations=nearmiss_cnt + oob_cnt + depleted_cnt,
            cfg=self.cfg,
        )

        self.prev_sinr_linear = sinr_lin.copy()
        self.prev_tx_powers = p_mat.copy()
        self.prev_channel_gains = radio_out["g_uav"].copy()

        self.step_idx += 1
        all_complete = bool(np.all(self.task_progress >= self.task_req))
        all_depleted = bool(np.all(self.uav_energy_j <= 0.0))
        horizon_reached = self.step_idx >= self.cfg.max_steps
        done = all_complete or all_depleted or horizon_reached

        next_actor_obs, next_critic_state = self._build_observations(
            comm_noise_std=self.pb_cfg.comm_noise_std
        )
        info: Dict[str, Any] = {
            "step": self.step_idx,
            "all_complete": all_complete,
            "horizon_reached": horizon_reached,
            "true_terminal": done,
            "reward_terms": reward_terms,
            "power_terms": power_terms,
        }
        if done:
            info["episode_metrics"] = self.get_episode_metrics()
        return next_actor_obs, next_critic_state, reward, done, info

    def get_episode_metrics(self) -> Dict[str, float]:
        base = super().get_episode_metrics()
        sub_denom = max(1, self.total_ctrl_substeps)
        tracking_rmse = float(np.sqrt(self.tracking_sq_err_sum / sub_denom))
        sat_rate = float(self.saturated_substeps) / float(sub_denom)
        att_rms_deg = float(np.degrees(np.sqrt(self.att_sq_sum / (2 * sub_denom))))
        norm_prog_sum = float(np.sum(np.minimum(self.task_progress, self.task_req) / self.task_req))
        utp = self.motor_energy_index_accum / (norm_prog_sum + self.pb_cfg.eps_utp)

        comp_diag = self.compensator.get_offline_diagnostics()
        base.update(
            {
                "motor_energy_index": float(self.motor_energy_index_accum),
                "tracking_rmse": tracking_rmse,
                "saturation_rate": sat_rate,
                "attitude_rms_deg": att_rms_deg,
                "utp": float(utp),
                "prediction_rmse": comp_diag["prediction_rmse"],
                "estimation_rmse_vs_r0": comp_diag["estimation_rmse_vs_r0"],
                "extra_comp_time_ms": comp_diag["extra_comp_time_ms"],
                "s_spec": compute_specialization_index(
                    self.routed_service_accum, eps_log=self.cfg.eps_log
                ),
                "s13_alignment": compute_s13_contribution_alignment(
                    self.routed_service_accum, self.uav_caps, self.initial_task_demands
                ),
            }
        )
        return base


def run_single_platform_tracking_trial(
    platform_type: str,
    controller_group: str = "A",
    seed: int = 0,
    total_duration_s: float = 150.0,
    dist_start_s: float = 30.0,
    dist_end_s: float = 90.0,
    dist_accel_mps2: float = 1.5,
    v_ref_mps: float = 3.0,
    recovery_tol_mps: float = 0.10,
    recovery_hold_s: float = 5.0,
) -> Dict[str, Any]:
    """Run Section 7.3 / Supplementary Table S53 single-platform 100-Hz tracking & recovery test.

    - Single quadrotor of `platform_type` ('fast', 'sensing', 'relay', 'endurance', 'general')
      with no high-level policy or communication.
    - Tracks constant horizontal velocity reference v_ref = 3.0 m/s.
    - Fixed-direction 1.5 m/s^2 horizontal acceleration disturbance active over [30, 90] s
      within a 150-s flight window (or scaled window for fast smoke tests).
    - Computes RMSE & max horizontal speed error norm over [dist_start_s, dist_end_s],
      and recovery time from dist_end_s to completion of first continuous recovery_hold_s
      window satisfying e_v(t) <= recovery_tol_mps (including the confirmation window).
    """
    if platform_type not in PLATFORM_SPECS:
        raise ValueError(f"Unknown platform_type: {platform_type}")

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    spec = PLATFORM_SPECS[platform_type]
    mass_i, _emax, _vmax, a_max_i, _xi, _g = spec

    dt = 0.01  # 100 Hz
    n_steps = int(round(total_duration_s / dt))
    k_start = int(round(dist_start_s / dt))
    k_end = int(round(dist_end_s / dt))
    hold_steps = int(round(recovery_hold_s / dt))

    # Matched initial velocity perturbation & fixed disturbance direction
    angle_ref = float(rng.uniform(0.0, 2.0 * np.pi))
    angle_dist = float(rng.uniform(0.0, 2.0 * np.pi))
    v_ref = np.array([v_ref_mps * np.cos(angle_ref), v_ref_mps * np.sin(angle_ref)], dtype=np.float64)
    d_vec = np.array(
        [dist_accel_mps2 * np.cos(angle_dist), dist_accel_mps2 * np.sin(angle_dist)],
        dtype=np.float64,
    )
    init_v_jitter = rng.normal(0.0, 0.08, size=2)

    pb_cfg = PyBulletConfig(gui=False)
    comp_cfg = CompensatorConfig(
        group=controller_group,  # 'A' for PID, 'C' for PID+PC
        dt_ctrl=dt,
        disturbance_mode="step_1p5",
    )
    compensator = DisturbanceCompensator(num_uavs=1, cfg=comp_cfg)

    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0.0, 0.0, -pb_cfg.gravity, physicsClientId=cid)
        p.setTimeStep(dt, physicsClientId=cid)
        uav_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.18, 0.18, 0.06], physicsClientId=cid
        )
        bid = p.createMultiBody(
            baseMass=mass_i,
            baseCollisionShapeIndex=uav_col,
            basePosition=[0.0, 0.0, pb_cfg.nominal_altitude_m],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=cid,
        )
        p.changeDynamics(
            bid,
            -1,
            localInertiaDiagonal=[0.012 * mass_i, 0.012 * mass_i, 0.022 * mass_i],
            linearDamping=0.0,
            angularDamping=0.04,
            physicsClientId=cid,
        )
        p.resetBaseVelocity(
            bid,
            linearVelocity=[float(v_ref[0] + init_v_jitter[0]), float(v_ref[1] + init_v_jitter[1]), 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=cid,
        )

        vel_int = np.zeros(2, dtype=np.float64)
        prev_vel_xy = v_ref + init_v_jitter
        thrust_act = mass_i * pb_cfg.gravity
        torque_act = np.zeros(3, dtype=np.float64)
        alpha_motor = dt / (pb_cfg.tau_motor + dt)

        err_trace = np.zeros(n_steps, dtype=np.float64)
        sat_count = 0

        for k in range(n_steps):
            pos_w, orn_w = p.getBasePositionAndOrientation(bid, physicsClientId=cid)
            lin_w, ang_w = p.getBaseVelocity(bid, physicsClientId=cid)
            vel_xy = np.array([lin_w[0], lin_w[1]], dtype=np.float64)
            euler = p.getEulerFromQuaternion(orn_w)
            rot_mat = np.asarray(p.getMatrixFromQuaternion(orn_w), dtype=np.float64).reshape(3, 3)
            ang_b = rot_mat.T @ np.asarray(ang_w, dtype=np.float64)

            e_v_vec = v_ref - vel_xy
            err_trace[k] = float(np.linalg.norm(e_v_vec))

            d_true_k = d_vec if (k_start <= k < k_end) else np.zeros(2, dtype=np.float64)
            a_meas = ((vel_xy - prev_vel_xy) / dt + rng.normal(0.0, 0.015, size=2))[None, :]
            prev_vel_xy = vel_xy.copy()

            u_ff = compensator.update(
                a_meas=a_meas,
                d_true_offline=d_true_k[None, :],
                pred_noise=np.zeros((1, 2), dtype=np.float64),
            )[0]

            # Baseline PID horizontal velocity tracking (modest integral limit so PID has realistic steady offset)
            vel_int = np.clip(vel_int + e_v_vec * dt, -1.6, 1.6)
            a_pid = pb_cfg.kp_pos * e_v_vec + 0.35 * vel_int
            a_des = a_pid + u_ff

            a_lim_horiz = min(pb_cfg.gravity * np.tan(pb_cfg.max_tilt_rad), 1.5 * a_max_i)
            a_norm = float(np.linalg.norm(a_des))
            sat_k = False
            if a_norm > a_lim_horiz:
                a_des = a_des * (a_lim_horiz / max(a_norm, 1e-12))
                sat_k = True

            roll_des = float(
                np.clip(-a_des[1] / pb_cfg.gravity, -pb_cfg.max_tilt_rad, pb_cfg.max_tilt_rad)
            )
            pitch_des = float(
                np.clip(a_des[0] / pb_cfg.gravity, -pb_cfg.max_tilt_rad, pb_cfg.max_tilt_rad)
            )
            e_z = pb_cfg.nominal_altitude_m - float(pos_w[2])
            a_z = 4.0 * e_z - 2.8 * float(lin_w[2])
            cos_tilt = max(0.5, float(np.cos(euler[0]) * np.cos(euler[1])))
            thrust_raw = mass_i * max(0.0, pb_cfg.gravity + a_z) / cos_tilt
            thrust_max = pb_cfg.thrust_margin_ratio * mass_i * pb_cfg.gravity
            if thrust_raw > thrust_max or thrust_raw <= 0.0:
                sat_k = True
            thrust_cmd = float(np.clip(thrust_raw, 0.0, thrust_max))
            sat_count += int(sat_k)

            att_err = np.array([roll_des - euler[0], pitch_des - euler[1], -euler[2]])
            att_err[2] = (att_err[2] + np.pi) % (2.0 * np.pi) - np.pi
            torque_cmd = (2.2 * mass_i) * (pb_cfg.kp_att * att_err - pb_cfg.kd_att * ang_b)
            torque_cmd = np.clip(torque_cmd, -1.5 * mass_i, 1.5 * mass_i)

            thrust_act = (1.0 - alpha_motor) * thrust_act + alpha_motor * thrust_cmd
            torque_act = (1.0 - alpha_motor) * torque_act + alpha_motor * torque_cmd

            u_hat = ((thrust_act / mass_i) * rot_mat[:2, 2])[None, :].astype(np.float64)
            compensator.set_applied_actuator_accel(u_hat)

            p.applyExternalForce(
                bid, -1, [0.0, 0.0, float(thrust_act)], [0.0, 0.0, 0.0], p.LINK_FRAME, physicsClientId=cid
            )
            p.applyExternalTorque(
                bid, -1, torque_act.tolist(), p.LINK_FRAME, physicsClientId=cid
            )
            if k_start <= k < k_end:
                p.applyExternalForce(
                    bid,
                    -1,
                    [float(mass_i * d_true_k[0]), float(mass_i * d_true_k[1]), 0.0],
                    list(pos_w),
                    p.WORLD_FRAME,
                    physicsClientId=cid,
                )
            p.stepSimulation(physicsClientId=cid)

        dist_slice = err_trace[k_start:k_end]
        rmse_dist = float(np.sqrt(np.mean(dist_slice ** 2))) if dist_slice.size > 0 else 0.0
        max_dist = float(np.max(dist_slice)) if dist_slice.size > 0 else 0.0

        # Evaluate continuous 5-s confirmation window after k_end (including hold duration)
        consec = 0
        recovered = False
        recovery_time_s = float("nan")
        for idx in range(k_end, n_steps):
            if err_trace[idx] <= recovery_tol_mps:
                consec += 1
                if consec >= hold_steps:
                    recovered = True
                    recovery_time_s = float((idx - k_end + 1) * dt)
                    break
            else:
                consec = 0

        return {
            "platform_type": platform_type,
            "controller": "PID" if controller_group.upper() == "A" else "PID+PC",
            "rmse_mps": rmse_dist,
            "max_err_mps": max_dist,
            "recovered": recovered,
            "recovery_time_s": recovery_time_s,
            "saturation_fraction": float(sat_count) / float(max(1, n_steps)),
            "error_trace_1hz": err_trace[::100].copy(),
        }
    finally:
        p.disconnect(physicsClientId=cid)
