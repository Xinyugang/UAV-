from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple
import numpy as np

from configs.defaults import (
    CAP_NORM_REF,
    EnvConfig,
    HELD_OUT_DEMAND_PROFILES,
    HOMOGENEOUS_SPEC,
    NOMINAL_TASK_PROFILES,
    PLATFORM_SPECS,
    PLATFORM_TYPES,
    S13_DESCRIPTOR_SHIFTS,
    S29_DEMAND_CONDITIONS,
    TASK_TYPES,
)
from envs.service_and_reward import (
    compute_routing_weights,
    compute_s13_contribution_alignment,
    compute_service_quality,
    compute_specialization_index,
    compute_step_energy,
    compute_step_reward,
)
from envs.wireless import (
    BeaconChannelBuffer,
    BeaconRecord,
    CandidateSelectionResult,
    StochasticWirelessChannel,
    deduplicate_and_select_candidates,
)


def _sample_points_with_min_dist(
    count: int,
    low: float,
    high: float,
    min_self_dist: float,
    other_groups: List[Tuple[np.ndarray, float]],
    rng: np.random.Generator,
    max_attempts: int = 5000,
) -> np.ndarray:
    """Rejection-sample 2D coordinates in [low, high]^2 satisfying pairwise clearance rules."""
    pts: List[np.ndarray] = []
    for _ in range(count):
        placed = False
        for _attempt in range(max_attempts):
            cand = rng.uniform(low, high, size=2)
            ok = True
            for existing in pts:
                if float(np.linalg.norm(cand - existing)) < min_self_dist:
                    ok = False
                    break
            if not ok:
                continue
            for grp_pts, grp_min_d in other_groups:
                if grp_pts.shape[0] > 0:
                    d_min = float(np.min(np.linalg.norm(grp_pts - cand[None, :], axis=-1)))
                    if d_min < grp_min_d:
                        ok = False
                        break
            if ok:
                pts.append(cand)
                placed = True
                break
        if not placed:
            # Fallback with relaxed self-clearance if arena is extraordinarily dense
            pts.append(rng.uniform(low, high, size=2))
    return np.asarray(pts, dtype=np.float64)


class LowFidelityUAVEnv:
    """Low-Fidelity 2D Heterogeneous Multi-UAV Dec-POMDP Environment.

    Implements Manuscript Section 3 (Eqs. 1-21, 23, 41-42) and Supplementary protocols:
      - LF & E1 stochastic LOS/NLOS/Rician channels (Eqs. 7-12, Table S26)
      - ADD / GEO / HARM & E7 service aggregation rules (Eqs. 14-19, Table S17)
      - Softmax & 20-m alternative service routing (Eq. 23, Table S28)
      - E3 beacon FIFO delay & packet loss, E4 multi-source ground interference (Table S26)
      - E5 capability shifts (Table S13), S4 demand shifts, S9 scene layout shifts
      - E-HET homogeneous fleet control (Table S25) & N=5/10/20 team-size scaling (S14, S51)
      - Exact state snapshot clone/restore for counterfactual & masking diagnostics (S44-S46)
    """

    def __init__(self, cfg: EnvConfig | None = None, training: bool = False) -> None:
        self.cfg = copy.deepcopy(cfg) if cfg is not None else EnvConfig()
        self.training = training

        self.rng = np.random.Generator(np.random.PCG64(0))
        self.wireless_channel = StochasticWirelessChannel(
            self.cfg.wireless, workspace_size=self.cfg.workspace_size
        )
        self.beacon_buffer = BeaconChannelBuffer(self.cfg.num_uavs, self.cfg.wireless)

        # Episode state arrays
        self.step_idx: int = 0
        self.uav_types: List[str] = []
        self.uav_roles: np.ndarray = np.zeros(self.cfg.num_uavs, dtype=np.int64)
        self.uav_caps: np.ndarray = np.zeros((self.cfg.num_uavs, 6), dtype=np.float64)
        self.uav_pos: np.ndarray = np.zeros((self.cfg.num_uavs, 2), dtype=np.float64)
        self.uav_vel: np.ndarray = np.zeros((self.cfg.num_uavs, 2), dtype=np.float64)
        self.uav_heading: np.ndarray = np.zeros(self.cfg.num_uavs, dtype=np.float64)
        self.uav_energy_j: np.ndarray = np.zeros(self.cfg.num_uavs, dtype=np.float64)

        self.task_centers: np.ndarray = np.zeros((self.cfg.num_tasks, 2), dtype=np.float64)
        self.task_demands: np.ndarray = np.zeros((self.cfg.num_tasks, 5), dtype=np.float64)
        self.initial_task_demands: np.ndarray = np.zeros((self.cfg.num_tasks, 5), dtype=np.float64)
        self.task_progress: np.ndarray = np.zeros(self.cfg.num_tasks, dtype=np.float64)
        self.task_req: np.ndarray = np.full(
            self.cfg.num_tasks, self.cfg.task_req_progress, dtype=np.float64
        )
        self.obstacle_centers: np.ndarray = np.zeros((self.cfg.num_obstacles, 2), dtype=np.float64)

        # Previous-step linear SINR matrix (lagged to t-1, initialized to 0.0 per Table S2)
        self.prev_sinr_linear: np.ndarray = np.zeros(
            (self.cfg.num_uavs, self.cfg.num_uavs), dtype=np.float64
        )
        self.prev_tx_powers: np.ndarray = np.zeros(
            (self.cfg.num_uavs, self.cfg.num_uavs), dtype=np.float64
        )
        self.prev_channel_gains: np.ndarray = np.zeros(
            (self.cfg.num_uavs, self.cfg.num_uavs), dtype=np.float64
        )
        self.latest_candidates: List[CandidateSelectionResult] = []

        # Episode accumulators for evaluation metrics
        self.routed_service_accum: np.ndarray = np.zeros(
            (self.cfg.num_uavs, self.cfg.num_tasks), dtype=np.float64
        )
        self.total_energy_j: float = 0.0
        self.total_collisions: int = 0
        self.total_uav_uav_collisions: int = 0
        self.total_uav_obs_collisions: int = 0
        self.total_nearmisses: int = 0
        self.total_oob: int = 0
        self.total_energy_depleted: int = 0
        self.attempted_links_total: int = 0
        self.effective_links_total: int = 0
        self.attempted_sinr_db_sum: float = 0.0
        self.candidate_edges_total: int = 0

    def _configure_fleet(self) -> None:
        """Assign platform types and 6D descriptors z_i^LF (Table 1, S13, S14, S25)."""
        n = self.cfg.num_uavs
        self.uav_types = [PLATFORM_TYPES[i % len(PLATFORM_TYPES)] for i in range(n)]
        self.uav_roles = np.array([i % len(PLATFORM_TYPES) for i in range(n)], dtype=np.int64)

        if self.cfg.homogeneous_team:
            self.uav_caps = np.tile(np.array(HOMOGENEOUS_SPEC, dtype=np.float64), (n, 1))
        else:
            caps = [list(PLATFORM_SPECS[t]) for t in self.uav_types]
            if self.cfg.s13_shift is not None:
                if self.cfg.s13_shift not in S13_DESCRIPTOR_SHIFTS:
                    raise ValueError(f"Unknown S13 shift: {self.cfg.s13_shift}")
                target_type, new_desc = S13_DESCRIPTOR_SHIFTS[self.cfg.s13_shift]
                for idx, t_name in enumerate(self.uav_types):
                    if t_name == target_type:
                        caps[idx] = list(new_desc)
            self.uav_caps = np.asarray(caps, dtype=np.float64)

        # Full battery capacity at episode start: E_i^0 = 1000 * E_i^max [J]
        self.uav_energy_j = self.uav_caps[:, 1] * 1000.0

    def _sample_scene_geometry(self) -> None:
        """Sample parent layout and optional S9 scene-layout shifts (Supplementary S9)."""
        n = self.cfg.num_uavs
        k_tasks = self.cfg.num_tasks
        n_obs_base = self.cfg.num_obstacles

        # Parent layout in [0, 100]^2
        tasks = _sample_points_with_min_dist(k_tasks, 10.0, 90.0, 20.0, [], self.rng)
        starts = _sample_points_with_min_dist(
            n, 5.0, 95.0, 3.0, [(tasks, 5.0)], self.rng
        )
        obs = _sample_points_with_min_dist(
            n_obs_base, 8.0, 92.0, 2.0, [(tasks, 3.0), (starts, 2.0)], self.rng
        )

        layout = self.cfg.scene_layout.lower()
        if layout in ("nominal", "in_distribution"):
            self.task_centers = tasks
            self.uav_pos = starts
            self.obstacle_centers = obs
        elif layout == "moderately_shifted":
            # S9: starts fixed; displace tasks by U[10, 16] m and obstacles by U[6, 10] m
            self.uav_pos = starts
            accepted = False
            for _ in range(2000):
                t_ang = self.rng.uniform(0.0, 2.0 * np.pi, size=k_tasks)
                t_len = self.rng.uniform(10.0, 16.0, size=k_tasks)
                t_new = tasks + np.stack([t_len * np.cos(t_ang), t_len * np.sin(t_ang)], axis=-1)

                o_ang = self.rng.uniform(0.0, 2.0 * np.pi, size=n_obs_base)
                o_len = self.rng.uniform(6.0, 10.0, size=n_obs_base)
                o_new = obs + np.stack([o_len * np.cos(o_ang), o_len * np.sin(o_ang)], axis=-1)

                if np.any(t_new < 5.0) or np.any(t_new > 95.0):
                    continue
                if np.any(o_new < 5.0) or np.any(o_new > 95.0):
                    continue
                # Check separations
                ok = True
                for a in range(k_tasks):
                    for b in range(a + 1, k_tasks):
                        if float(np.linalg.norm(t_new[a] - t_new[b])) < 20.0:
                            ok = False
                            break
                if not ok:
                    continue
                if np.min(np.linalg.norm(starts[:, None, :] - t_new[None, :, :], axis=-1)) < 5.0:
                    continue
                for a in range(n_obs_base):
                    for b in range(a + 1, n_obs_base):
                        if float(np.linalg.norm(o_new[a] - o_new[b])) < 2.0:
                            ok = False
                            break
                if not ok:
                    continue
                if n_obs_base > 0:
                    if np.min(np.linalg.norm(o_new[:, None, :] - t_new[None, :, :], axis=-1)) < 3.0:
                        continue
                    if np.min(np.linalg.norm(o_new[:, None, :] - starts[None, :, :], axis=-1)) < 2.0:
                        continue
                self.task_centers = t_new
                self.obstacle_centers = o_new
                accepted = True
                break
            if not accepted:
                self.task_centers = np.clip(tasks + 8.0, 10.0, 90.0)
                self.obstacle_centers = obs
        elif layout == "more_obstructed":
            # S9: keep parent layout (5 obstacles) and add 3 corridor obstacles (total 8)
            self.task_centers = tasks
            self.uav_pos = starts
            obs_list = [o.copy() for o in obs]
            for k in range(k_tasks):
                d_to_k = np.linalg.norm(starts - tasks[k][None, :], axis=-1)
                nearest_s = starts[int(np.argmin(d_to_k))]
                seg = tasks[k] - nearest_s
                seg_len = max(float(np.linalg.norm(seg)), 1e-6)
                u_dir = seg / seg_len
                p_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float64)

                placed_extra = False
                for _ in range(1000):
                    frac = float(self.rng.uniform(0.35, 0.65))
                    perp = float(self.rng.uniform(-0.35, 0.35))
                    cand = nearest_s + frac * seg + perp * p_dir
                    if np.any(cand < 5.0) or np.any(cand > 95.0):
                        continue
                    if float(np.min(np.linalg.norm(tasks - cand[None, :], axis=-1))) < 3.0:
                        continue
                    if float(np.min(np.linalg.norm(starts - cand[None, :], axis=-1))) < 2.0:
                        continue
                    if any(float(np.linalg.norm(cand - prev_o)) < 2.0 for prev_o in obs_list):
                        continue
                    obs_list.append(cand)
                    placed_extra = True
                    break
                if not placed_extra:
                    obs_list.append(0.5 * (nearest_s + tasks[k]))
            self.obstacle_centers = np.asarray(obs_list, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported scene_layout: {self.cfg.scene_layout}")

    def _sample_task_demands(self, demand_rng: np.random.Generator | None = None) -> None:
        """Sample task demand vectors d_k^t (Table 2, S4 Eq. S2, S29)."""
        gen = self.rng if demand_rng is None else demand_rng
        mode = self.cfg.demand_mode
        k_tasks = self.cfg.num_tasks

        if mode == "nominal":
            demands = [
                np.array(NOMINAL_TASK_PROFILES[TASK_TYPES[k % len(TASK_TYPES)]], dtype=np.float64)
                for k in range(k_tasks)
            ]
            arr = np.asarray(demands, dtype=np.float64)
            if self.training and self.cfg.train_demand_noise_std > 0.0:
                noise = gen.normal(0.0, self.cfg.train_demand_noise_std, size=(k_tasks, 4))
                arr[:, 1:5] = np.clip(arr[:, 1:5] + noise, 0.0, 1.0)
            self.task_demands = arr
        elif mode in HELD_OUT_DEMAND_PROFILES:
            vec = np.array(HELD_OUT_DEMAND_PROFILES[mode], dtype=np.float64)
            self.task_demands = np.tile(vec, (k_tasks, 1))
        elif mode == "continuous_random":
            # Eq. (S2): U[0.25, 0.95]^4 rejected unless Euclidean distance > 0.20 from templates
            templates = np.array(
                [NOMINAL_TASK_PROFILES[t][1:5] for t in TASK_TYPES], dtype=np.float64
            )
            rows = []
            for _ in range(k_tasks):
                while True:
                    w_prop = gen.uniform(0.25, 0.95, size=4)
                    min_dist = float(np.min(np.linalg.norm(templates - w_prop[None, :], axis=-1)))
                    if min_dist > 0.20:
                        rows.append(np.concatenate(([1.0], w_prop)))
                        break
            self.task_demands = np.asarray(rows, dtype=np.float64)
        elif mode in S29_DEMAND_CONDITIONS:
            cond_vecs = S29_DEMAND_CONDITIONS[mode]
            self.task_demands = np.asarray(
                [cond_vecs[k % len(cond_vecs)].copy() for k in range(k_tasks)], dtype=np.float64
            )
        else:
            raise ValueError(f"Unsupported demand_mode: {mode}")

        self.initial_task_demands = self.task_demands.copy()

    def _current_beacon_records(self) -> List[BeaconRecord]:
        t_sec = self.step_idx * self.cfg.dt
        return [
            BeaconRecord(
                sender_id=i,
                timestamp=t_sec,
                seq_num=self.step_idx * self.cfg.num_uavs + i,
                pos=self.uav_pos[i].copy(),
                vel=self.uav_vel[i].copy(),
                energy_j=float(self.uav_energy_j[i]),
                cap_vec=self.uav_caps[i].copy(),
            )
            for i in range(self.cfg.num_uavs)
        ]

    def reset(
        self,
        seed: int | None = None,
        demand_seed: int | None = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Reset episode and return (actor_obs_batch, critic_state_dict)."""
        if seed is not None:
            self.rng = np.random.Generator(np.random.PCG64(int(seed)))
        dem_rng = (
            np.random.Generator(np.random.PCG64(int(demand_seed)))
            if demand_seed is not None
            else self.rng
        )

        self.step_idx = 0
        self._configure_fleet()
        self._sample_scene_geometry()
        self._sample_task_demands(dem_rng)

        n = self.cfg.num_uavs
        self.uav_vel = np.zeros((n, 2), dtype=np.float64)
        self.uav_heading = np.zeros(n, dtype=np.float64)
        self.task_progress = np.zeros(self.cfg.num_tasks, dtype=np.float64)
        self.task_req = np.full(
            self.cfg.num_tasks, self.cfg.task_req_progress, dtype=np.float64
        )

        self.prev_sinr_linear = np.zeros((n, n), dtype=np.float64)
        self.prev_tx_powers = np.zeros((n, n), dtype=np.float64)
        self.prev_channel_gains = np.full((n, n), 1e-9, dtype=np.float64)
        np.fill_diagonal(self.prev_channel_gains, 0.0)

        self.routed_service_accum = np.zeros((n, self.cfg.num_tasks), dtype=np.float64)
        self.total_energy_j = 0.0
        self.total_collisions = 0
        self.total_uav_uav_collisions = 0
        self.total_uav_obs_collisions = 0
        self.total_nearmisses = 0
        self.total_oob = 0
        self.total_energy_depleted = 0
        self.attempted_links_total = 0
        self.effective_links_total = 0
        self.attempted_sinr_db_sum = 0.0
        self.candidate_edges_total = 0

        self.wireless_channel.reset_episode(self.uav_pos, self.obstacle_centers, self.rng)
        init_records = self._current_beacon_records()
        self.beacon_buffer.reset(init_records)
        return self._build_observations(comm_noise_std=0.0)

    def _build_observations(
        self, comm_noise_std: float = 0.0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Construct structured decentralized Actor observations (Eq. 2) and Critic state (Eq. 33)."""
        n = self.cfg.num_uavs
        m_cap = self.cfg.wireless.m_cap
        obs_cap = self.cfg.obs_cap
        cap_ref = np.asarray(CAP_NORM_REF, dtype=np.float64)

        records_now = self._current_beacon_records()
        received_lists = self.beacon_buffer.push_and_receive(
            records_now, self.rng, comm_noise_std=comm_noise_std
        )

        alpha_query, _, u_unfinished = compute_routing_weights(
            self.uav_pos,
            self.task_centers,
            self.task_progress,
            self.task_req,
            gamma_d=self.cfg.gamma_d,
            gamma_u=self.cfg.gamma_u,
            d_norm=self.cfg.d_norm,
            routing_mode="softmax",
        )

        # Normalized self state x_i^t = [p_x/100, p_y/100, v_x/8, v_y/8, E_i/150000] (5D)
        x_self = np.concatenate(
            [
                self.uav_pos / self.cfg.d_norm,
                self.uav_vel / self.cfg.v_max_ref,
                (self.uav_energy_j / self.cfg.e_max_ref_j)[:, None],
            ],
            axis=-1,
        )
        # Normalized capability z_i (6D; values > 1 NOT clipped per Table S26)
        z_self = self.uav_caps / cap_ref[None, :]
        # Weighted demand mixture D_i^t = sum_k alpha_ik^t * d_k^t (5D)
        demand_mix = alpha_query @ self.task_demands  # (N, 5)

        # All-task local observation features per UAV: K=3 tasks x 8 features = 24D
        # [rel_x/100, rel_y/100, d_ik/100, u_k, alpha_ik, nu_k, sigma_k, lam_k_or_eps] -> exact 8 features
        task_obs = np.zeros((n, self.cfg.num_tasks * 8), dtype=np.float64)
        for i in range(n):
            feats_k = []
            for k in range(self.cfg.num_tasks):
                rel = (self.task_centers[k] - self.uav_pos[i]) / self.cfg.d_norm
                d_ik = float(np.linalg.norm(rel))
                # 8 features per task: rel_x, rel_y, dist, unfinished u_k, alpha_ik, nu_k, sigma_k, 0.5*(eps_k+lam_k)
                dk = self.task_demands[k]
                feats_k.extend(
                    [
                        rel[0],
                        rel[1],
                        d_ik,
                        u_unfinished[k],
                        alpha_query[i, k],
                        dk[1],
                        dk[2],
                        0.5 * (dk[3] + dk[4]),
                    ]
                )
            task_obs[i] = np.asarray(feats_k, dtype=np.float64)

        # Nearest obs_cap=5 obstacles per UAV: 5 x 4 features = 20D
        # [rel_x/100, rel_y/100, center_dist/100, surface_gap/10]
        obs_feats = np.zeros((n, obs_cap * 4), dtype=np.float64)
        n_obs_actual = self.obstacle_centers.shape[0]
        for i in range(n):
            if n_obs_actual > 0:
                rel_all = self.obstacle_centers - self.uav_pos[i][None, :]
                d_all = np.linalg.norm(rel_all, axis=-1)
                order = np.argsort(d_all)[:obs_cap]
                for slot, o_idx in enumerate(order):
                    rx, ry = rel_all[o_idx] / self.cfg.d_norm
                    dc = d_all[o_idx] / self.cfg.d_norm
                    gap = (d_all[o_idx] - self.cfg.d_col_obs) / 10.0
                    obs_feats[i, slot * 4 : (slot + 1) * 4] = [rx, ry, dc, gap]

        # Candidate selection & padding for M_i^t (M_cap = 3)
        nbr_x = np.zeros((n, m_cap, 5), dtype=np.float64)
        nbr_z = np.zeros((n, m_cap, 6), dtype=np.float64)
        nbr_rel = np.zeros((n, m_cap, 7), dtype=np.float64)
        nbr_dist_cost = np.zeros((n, m_cap), dtype=np.float64)
        nbr_hist_cost = np.zeros((n, m_cap), dtype=np.float64)
        nbr_energy_cost = np.zeros((n, m_cap), dtype=np.float64)
        nbr_mask = np.zeros((n, m_cap), dtype=bool)
        nbr_ids = np.full((n, m_cap), -1, dtype=np.int64)

        self.latest_candidates = []
        for i in range(n):
            cand = deduplicate_and_select_candidates(
                self_id=i,
                self_pos=self.uav_pos[i],
                records=received_lists[i],
                d_comm=self.cfg.wireless.d_comm,
                m_cap=m_cap,
            )
            self.latest_candidates.append(cand)
            self.candidate_edges_total += cand.unique_count

            for slot, uid in enumerate(cand.valid_ids):
                nbr_mask[i, slot] = True
                nbr_ids[i, slot] = uid
                p_j = cand.padded_pos[slot]
                v_j = cand.padded_vel[slot]
                e_j = cand.padded_energy_j[slot]
                cap_j = cand.padded_cap_vec[slot]
                e_j_max_j = max(cap_j[1] * 1000.0, 1e-6)

                nbr_x[i, slot] = [
                    p_j[0] / self.cfg.d_norm,
                    p_j[1] / self.cfg.d_norm,
                    v_j[0] / self.cfg.v_max_ref,
                    v_j[1] / self.cfg.v_max_ref,
                    e_j / self.cfg.e_max_ref_j,
                ]
                nbr_z[i, slot] = cap_j / cap_ref

                # Eq. (26) dimensionless physical penalty terms:
                d_cost = cand.padded_dist[slot] / self.cfg.d_norm
                sinr_prev = float(self.prev_sinr_linear[i, uid])
                c_hist = max(0.0, 1.0 - sinr_prev / self.cfg.wireless.sinr_ref)
                c_ener = max(0.0, 1.0 - e_j / e_j_max_j)

                nbr_dist_cost[i, slot] = d_cost
                nbr_hist_cost[i, slot] = c_hist
                nbr_energy_cost[i, slot] = c_ener

                rel_p = (p_j - self.uav_pos[i]) / self.cfg.d_norm
                rel_v = (v_j - self.uav_vel[i]) / self.cfg.v_max_ref
                nbr_rel[i, slot] = [
                    rel_p[0],
                    rel_p[1],
                    rel_v[0],
                    rel_v[1],
                    d_cost,
                    c_hist,
                    c_ener,
                ]

        actor_obs = {
            "x_self": x_self.astype(np.float32),
            "z_self": z_self.astype(np.float32),
            "uav_roles": self.uav_roles.copy(),
            "demand_mix": demand_mix.astype(np.float32),
            "task_obs": task_obs.astype(np.float32),
            "obs_feats": obs_feats.astype(np.float32),
            "nbr_x": nbr_x.astype(np.float32),
            "nbr_z": nbr_z.astype(np.float32),
            "nbr_rel": nbr_rel.astype(np.float32),
            "nbr_dist_cost": nbr_dist_cost.astype(np.float32),
            "nbr_hist_cost": nbr_hist_cost.astype(np.float32),
            "nbr_energy_cost": nbr_energy_cost.astype(np.float32),
            "nbr_mask": nbr_mask,
            "nbr_ids": nbr_ids,
        }

        # Centralized Critic inputs (Eq. 33): S_UAV (N, 5), Z (N, 6), Tasks (K, 8), Obstacles (N_obs, 3)
        critic_uav = np.concatenate([x_self, z_self], axis=-1).astype(np.float32)  # (N, 11)
        critic_tasks = np.concatenate(
            [
                self.task_centers / self.cfg.d_norm,
                self.task_demands,
                u_unfinished[:, None],
            ],
            axis=-1,
        ).astype(np.float32)  # (K, 8)
        if n_obs_actual > 0:
            critic_obs = np.concatenate(
                [
                    self.obstacle_centers / self.cfg.d_norm,
                    np.full((n_obs_actual, 1), self.cfg.d_col_obs / 10.0),
                ],
                axis=-1,
            ).astype(np.float32)
        else:
            critic_obs = np.zeros((1, 3), dtype=np.float32)

        critic_state = {
            "uav_features": critic_uav,
            "task_features": critic_tasks,
            "obs_features": critic_obs,
        }
        return actor_obs, critic_state

    def apply_motion_constraints(self, raw_u_cmd: np.ndarray) -> np.ndarray:
        """Enforce platform speed v_i^max and acceleration a_i^max limits on planar velocity command."""
        u_cmd = np.asarray(raw_u_cmd, dtype=np.float64).copy()
        v_max = self.uav_caps[:, 2]
        a_max = self.uav_caps[:, 3]

        # Clip desired command speed to v_max
        cmd_norm = np.linalg.norm(u_cmd, axis=-1, keepdims=True)
        scale_cmd = np.where(cmd_norm > v_max[:, None], v_max[:, None] / np.maximum(cmd_norm, 1e-12), 1.0)
        u_clipped = u_cmd * scale_cmd

        # Enforce acceleration bound ||v^{t+1} - v^t||_2 <= a_i^max * dt
        dv = u_clipped - self.uav_vel
        dv_norm = np.linalg.norm(dv, axis=-1, keepdims=True)
        max_dv = (a_max * self.cfg.dt)[:, None]
        scale_dv = np.where(dv_norm > max_dv, max_dv / np.maximum(dv_norm, 1e-12), 1.0)
        v_next = self.uav_vel + dv * scale_dv

        # Final speed safety clip & zero velocity if battery depleted
        v_norm = np.linalg.norm(v_next, axis=-1, keepdims=True)
        scale_v = np.where(v_norm > v_max[:, None], v_max[:, None] / np.maximum(v_norm, 1e-12), 1.0)
        v_next = v_next * scale_v
        depleted = self.uav_energy_j <= 0.0
        v_next[depleted] = 0.0
        return v_next

    def build_power_matrix_from_slot_powers(self, slot_powers: np.ndarray) -> np.ndarray:
        """Map per-UAV padded slot powers (N, M_cap) onto full directed matrix (N, N)."""
        n = self.cfg.num_uavs
        p_mat = np.zeros((n, n), dtype=np.float64)
        for i, cand in enumerate(self.latest_candidates):
            if self.uav_energy_j[i] <= 0.0:
                continue
            p_slots = np.maximum(0.0, np.asarray(slot_powers[i], dtype=np.float64))
            # Enforce per-UAV budget projection sum_j P_ij <= P_max
            valid_p = np.zeros_like(p_slots)
            for s_idx, uid in enumerate(cand.valid_ids):
                valid_p[s_idx] = p_slots[s_idx]
            s_tot = float(np.sum(valid_p))
            if s_tot > self.cfg.wireless.p_max:
                valid_p *= self.cfg.wireless.p_max / s_tot
            for s_idx, uid in enumerate(cand.valid_ids):
                p_mat[i, uid] = valid_p[s_idx]
        return p_mat

    def step(
        self,
        motion_cmds: np.ndarray,
        slot_powers: np.ndarray,
        ground_power_override: float | None = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Advance environment one 0.2-s decision step (Algorithm S1 caller note & Algorithm S2)."""
        n = self.cfg.num_uavs
        # 1. Update planar kinematics
        v_exec = self.apply_motion_constraints(motion_cmds)
        p_raw = self.uav_pos + v_exec * self.cfg.dt

        oob_mask = (
            (p_raw[:, 0] < 0.0)
            | (p_raw[:, 0] > self.cfg.workspace_size)
            | (p_raw[:, 1] < 0.0)
            | (p_raw[:, 1] > self.cfg.workspace_size)
        )
        oob_cnt = int(np.sum(oob_mask))
        self.total_oob += oob_cnt

        self.uav_pos = np.clip(p_raw, 0.0, self.cfg.workspace_size)
        self.uav_vel = v_exec
        moving = np.linalg.norm(v_exec, axis=-1) > 1e-6
        self.uav_heading = np.where(
            moving, np.arctan2(v_exec[:, 1], v_exec[:, 0]), self.uav_heading
        )

        # 2. Evaluate wireless channel, co-channel + ground interference, and SINR (Eqs. 8-12)
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

        # Eq. (15): effective outgoing link ratio chi_i^t over retained neighbor set N_i^t
        chi_comm = np.zeros(n, dtype=np.float64)
        for i, cand in enumerate(self.latest_candidates):
            m_i = len(cand.valid_ids)
            eff_i = sum(int(effective[i, uid]) for uid in cand.valid_ids)
            chi_comm[i] = eff_i / (m_i + self.cfg.wireless.eps_comm)

        # 3. Routing & service progress update (Eqs. 13-19, 23)
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
        routed_q = alpha_env * q_mat  # (N, K)
        self.routed_service_accum += routed_q
        prev_prog = self.task_progress.copy()
        self.task_progress = self.task_progress + np.sum(routed_q, axis=0) * self.cfg.dt

        # 4. Collisions & near-misses (Table 3)
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

        # 5. Energy integration (Eq. 20, Table S26)
        delta_e_j, power_terms = compute_step_energy(
            self.uav_caps, self.uav_vel, p_mat, self.uav_pos, self.task_centers, self.cfg
        )
        self.uav_energy_j = np.maximum(0.0, self.uav_energy_j - delta_e_j)
        self.total_energy_j += float(np.sum(delta_e_j))
        depleted_cnt = int(np.sum(self.uav_energy_j <= 0.0))
        self.total_energy_depleted += int(depleted_cnt > 0)

        # 6. Reward (Eq. 21, Table S31)
        n_constraints = nearmiss_cnt + oob_cnt + depleted_cnt
        reward, reward_terms = compute_step_reward(
            prev_progress=prev_prog,
            new_progress=self.task_progress,
            task_req=self.task_req,
            delta_e_j=delta_e_j,
            n_collisions=step_collisions,
            chi_comm=chi_comm,
            n_constraint_violations=n_constraints,
            cfg=self.cfg,
        )

        # Update lagged wireless histories for t+1
        self.prev_sinr_linear = sinr_lin.copy()
        self.prev_tx_powers = p_mat.copy()
        self.prev_channel_gains = radio_out["g_uav"].copy()

        self.step_idx += 1
        all_complete = bool(np.all(self.task_progress >= self.task_req))
        all_depleted = bool(np.all(self.uav_energy_j <= 0.0))
        horizon_reached = self.step_idx >= self.cfg.max_steps
        done = all_complete or all_depleted or horizon_reached

        if all_complete:
            reward += 2.0  # Terminal full-completion bonus

        next_actor_obs, next_critic_state = self._build_observations(comm_noise_std=0.0)
        info: Dict[str, Any] = {
            "step": self.step_idx,
            "all_complete": all_complete,
            "horizon_reached": horizon_reached,
            "true_terminal": done,  # Per Eq. (39): 1500-step task-ending horizon is terminal (B_t=0)
            "attempted_step": n_att_step,
            "effective_step": n_eff_step,
            "collisions_step": step_collisions,
            "obs_collisions_step": uav_obs_col,
            "sinr_linear": sinr_lin,
            "sinr_db": sinr_db,
            "attempted_matrix": attempted,
            "effective_matrix": effective,
            "power_matrix": p_mat,
            "reward_terms": reward_terms,
            "power_terms": power_terms,
        }
        if done:
            info["episode_metrics"] = self.get_episode_metrics()
        return next_actor_obs, next_critic_state, reward, done, info

    def get_episode_metrics(self) -> Dict[str, float]:
        """Compute complete episode summary metrics (Section 5.2, Eqs. 20, 41-42, S3, S5-S6)."""
        steps = max(1, self.step_idx)
        comp_mask = self.task_progress >= self.task_req
        partial = float(np.mean(comp_mask))
        full = float(np.all(comp_mask))
        cf_full = float(bool(full > 0.5) and (self.total_collisions == 0))
        comm_succ = (
            float(self.effective_links_total)
            / (float(self.attempted_links_total) + self.cfg.wireless.eps_comm)
            if self.attempted_links_total > 0
            else 0.0
        )
        mean_sinr_db = (
            self.attempted_sinr_db_sum / float(self.attempted_links_total)
            if self.attempted_links_total > 0
            else float("nan")
        )
        return {
            "full_success": full,
            "cf_full_success": cf_full,
            "partial_completion": partial,
            "team_energy_kj": self.total_energy_j * 1e-3,
            "comm_success": comm_succ,
            "mean_sinr_db": mean_sinr_db,
            "collision_rate": float(self.total_collisions) / float(steps),
            "obs_events_per_frame": float(self.total_uav_obs_collisions) / float(steps),
            "nearmiss_rate": float(self.total_nearmisses) / float(steps),
            "s_spec": compute_specialization_index(
                self.routed_service_accum, eps_log=self.cfg.eps_log
            ),
            "s13_alignment": compute_s13_contribution_alignment(
                self.routed_service_accum, self.uav_caps, self.initial_task_demands
            ),
            "attempted_links_total": float(self.attempted_links_total),
            "effective_links_total": float(self.effective_links_total),
            "attempted_per_frame": float(self.attempted_links_total) / float(steps),
            "effective_per_frame": float(self.effective_links_total) / float(steps),
            "candidate_edges_per_frame": float(self.candidate_edges_total) / float(steps),
            "executed_steps": float(steps),
            "oob_events": float(self.total_oob),
            "energy_depleted_frames": float(self.total_energy_depleted),
        }

    def clone_state(self) -> Dict[str, Any]:
        """Snapshot complete simulator and RNG state for counterfactual branching (S44-S46)."""
        return {
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "step_idx": self.step_idx,
            "uav_types": list(self.uav_types),
            "uav_roles": self.uav_roles.copy(),
            "uav_caps": self.uav_caps.copy(),
            "uav_pos": self.uav_pos.copy(),
            "uav_vel": self.uav_vel.copy(),
            "uav_heading": self.uav_heading.copy(),
            "uav_energy_j": self.uav_energy_j.copy(),
            "task_centers": self.task_centers.copy(),
            "task_demands": self.task_demands.copy(),
            "initial_task_demands": self.initial_task_demands.copy(),
            "task_progress": self.task_progress.copy(),
            "task_req": self.task_req.copy(),
            "obstacle_centers": self.obstacle_centers.copy(),
            "prev_sinr_linear": self.prev_sinr_linear.copy(),
            "prev_tx_powers": self.prev_tx_powers.copy(),
            "prev_channel_gains": self.prev_channel_gains.copy(),
            "latest_candidates": copy.deepcopy(self.latest_candidates),
            "routed_service_accum": self.routed_service_accum.copy(),
            "total_energy_j": self.total_energy_j,
            "total_collisions": self.total_collisions,
            "total_uav_uav_collisions": self.total_uav_uav_collisions,
            "total_uav_obs_collisions": self.total_uav_obs_collisions,
            "total_nearmisses": self.total_nearmisses,
            "total_oob": self.total_oob,
            "total_energy_depleted": self.total_energy_depleted,
            "attempted_links_total": self.attempted_links_total,
            "effective_links_total": self.effective_links_total,
            "attempted_sinr_db_sum": self.attempted_sinr_db_sum,
            "candidate_edges_total": self.candidate_edges_total,
            "uav_los_matrix": None
            if self.wireless_channel.uav_los_matrix is None
            else self.wireless_channel.uav_los_matrix.copy(),
            "ground_los_matrix": None
            if self.wireless_channel.ground_los_matrix is None
            else self.wireless_channel.ground_los_matrix.copy(),
            "ground_positions": self.wireless_channel.ground_positions.copy(),
            "beacon_buffer": copy.deepcopy(self.beacon_buffer),
        }

    def restore_state(self, snap: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Restore complete simulator and RNG state from snapshot."""
        self.rng.bit_generator.state = copy.deepcopy(snap["rng_state"])
        self.step_idx = int(snap["step_idx"])
        self.uav_types = list(snap["uav_types"])
        self.uav_roles = snap["uav_roles"].copy()
        self.uav_caps = snap["uav_caps"].copy()
        self.uav_pos = snap["uav_pos"].copy()
        self.uav_vel = snap["uav_vel"].copy()
        self.uav_heading = snap["uav_heading"].copy()
        self.uav_energy_j = snap["uav_energy_j"].copy()
        self.task_centers = snap["task_centers"].copy()
        self.task_demands = snap["task_demands"].copy()
        self.initial_task_demands = snap["initial_task_demands"].copy()
        self.task_progress = snap["task_progress"].copy()
        self.task_req = snap["task_req"].copy()
        self.obstacle_centers = snap["obstacle_centers"].copy()
        self.prev_sinr_linear = snap["prev_sinr_linear"].copy()
        self.prev_tx_powers = snap["prev_tx_powers"].copy()
        self.prev_channel_gains = snap["prev_channel_gains"].copy()
        self.latest_candidates = copy.deepcopy(snap["latest_candidates"])
        self.routed_service_accum = snap["routed_service_accum"].copy()
        self.total_energy_j = float(snap["total_energy_j"])
        self.total_collisions = int(snap["total_collisions"])
        self.total_uav_uav_collisions = int(snap["total_uav_uav_collisions"])
        self.total_uav_obs_collisions = int(snap["total_uav_obs_collisions"])
        self.total_nearmisses = int(snap["total_nearmisses"])
        self.total_oob = int(snap["total_oob"])
        self.total_energy_depleted = int(snap["total_energy_depleted"])
        self.attempted_links_total = int(snap["attempted_links_total"])
        self.effective_links_total = int(snap["effective_links_total"])
        self.attempted_sinr_db_sum = float(snap["attempted_sinr_db_sum"])
        self.candidate_edges_total = int(snap["candidate_edges_total"])
        self.wireless_channel.uav_los_matrix = (
            None if snap["uav_los_matrix"] is None else snap["uav_los_matrix"].copy()
        )
        self.wireless_channel.ground_los_matrix = (
            None if snap["ground_los_matrix"] is None else snap["ground_los_matrix"].copy()
        )
        self.wireless_channel.ground_positions = snap["ground_positions"].copy()
        self.beacon_buffer = copy.deepcopy(snap["beacon_buffer"])
        return self._build_observations(comm_noise_std=0.0)
