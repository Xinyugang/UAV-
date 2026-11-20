from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence, Tuple
import numpy as np

from configs.defaults import WirelessConfig


@dataclass
class BeaconRecord:
    """Summary beacon broadcast by a UAV for neighbor discovery (Eq. 2, Table S16)."""
    sender_id: int
    timestamp: float
    seq_num: int
    pos: np.ndarray        # shape (2,) [m]
    vel: np.ndarray        # shape (2,) [m/s]
    energy_j: float        # available energy [J]
    cap_vec: np.ndarray    # shape (6,): [m, E_max_kJ, v_max, a_max, xi_sen, g]


@dataclass
class CandidateSelectionResult:
    """Ordered, deduplicated, truncated neighbor set M_i^t (Algorithm S1 Step 1)."""
    valid_ids: List[int]           # Distinct retained neighbor IDs (len M <= M_cap)
    valid_mask: np.ndarray         # shape (M_cap,) bool
    padded_pos: np.ndarray         # shape (M_cap, 2)
    padded_vel: np.ndarray         # shape (M_cap, 2)
    padded_energy_j: np.ndarray    # shape (M_cap,)
    padded_cap_vec: np.ndarray     # shape (M_cap, 6)
    padded_dist: np.ndarray        # shape (M_cap,)
    unique_count: int              # M_unique <= N - 1 before M_cap truncation (Table S16)
    raw_record_count: int          # Total records inspected before self-filter & dedup


def deduplicate_and_select_candidates(
    self_id: int,
    self_pos: np.ndarray,
    records: Sequence[BeaconRecord],
    d_comm: float = 50.0,
    m_cap: int = 3,
) -> CandidateSelectionResult:
    """Filter self-records, deduplicate by UAV ID, filter by d_comm, and keep nearest M_cap.

    Follows Manuscript Eq. (7), Algorithm S1 Step 1, and Supplementary Table S16:
      1. Exclude records with sender_id == self_id.
      2. Deduplicate by sender_id: retain latest timestamp, breaking ties by largest seq_num.
      3. Filter to candidates within Euclidean distance d_ij <= d_comm.
      4. Sort by (d_ij, sender_id) ascending and retain at most M_cap candidates.
      5. Pad arrays to M_cap and record valid identities and boolean mask.
    """
    latest_by_id: Dict[int, BeaconRecord] = {}
    for rec in records:
        if rec.sender_id == self_id:
            continue
        prev = latest_by_id.get(rec.sender_id)
        if prev is None or (rec.timestamp, rec.seq_num) > (prev.timestamp, prev.seq_num):
            latest_by_id[rec.sender_id] = rec

    candidates_in_range: List[Tuple[float, int, BeaconRecord]] = []
    p_i = np.asarray(self_pos, dtype=np.float64)
    for uid, rec in latest_by_id.items():
        d_ij = float(np.linalg.norm(p_i - np.asarray(rec.pos, dtype=np.float64)))
        if d_ij <= d_comm:
            candidates_in_range.append((d_ij, uid, rec))

    unique_count = len(candidates_in_range)
    candidates_in_range.sort(key=lambda item: (item[0], item[1]))
    retained = candidates_in_range[:m_cap]

    valid_ids: List[int] = []
    valid_mask = np.zeros(m_cap, dtype=bool)
    padded_pos = np.zeros((m_cap, 2), dtype=np.float64)
    padded_vel = np.zeros((m_cap, 2), dtype=np.float64)
    padded_energy_j = np.zeros(m_cap, dtype=np.float64)
    padded_cap_vec = np.zeros((m_cap, 6), dtype=np.float64)
    padded_dist = np.zeros(m_cap, dtype=np.float64)

    for slot, (dist, uid, rec) in enumerate(retained):
        valid_ids.append(uid)
        valid_mask[slot] = True
        padded_pos[slot] = rec.pos
        padded_vel[slot] = rec.vel
        padded_energy_j[slot] = rec.energy_j
        padded_cap_vec[slot] = rec.cap_vec
        padded_dist[slot] = dist

    return CandidateSelectionResult(
        valid_ids=valid_ids,
        valid_mask=valid_mask,
        padded_pos=padded_pos,
        padded_vel=padded_vel,
        padded_energy_j=padded_energy_j,
        padded_cap_vec=padded_cap_vec,
        padded_dist=padded_dist,
        unique_count=unique_count,
        raw_record_count=len(records),
    )


class BeaconChannelBuffer:
    """FIFO beacon delay and whole-beacon packet-loss observation kernel (E3, Table S26).

    - 200 ms policy step corresponds to 5 slots of 40 ms each.
    - When delay_slots == 0 and loss_prob == 0.0 (nominal training/eval), returns exact
      instantaneous summaries with zero latency and zero loss.
    - When delay_slots > 0 or loss_prob > 0.0 (zero-shot E3 tests), buffers 5 sub-slots per
      policy step, applies Bernoulli(loss_prob) whole-beacon drop per receiver-sender pair,
      and applies zero-order hold (cached latest received summary) on misses.
    """

    def __init__(self, num_uavs: int, cfg: WirelessConfig) -> None:
        self.num_uavs = num_uavs
        self.cfg = cfg
        self.slots_per_step = max(1, int(round(200.0 / cfg.beacon_slot_ms)))
        self._fifo: Deque[List[BeaconRecord]] = deque()
        self._receiver_cache: Dict[Tuple[int, int], BeaconRecord] = {}
        self._seq_counter: int = 0

    def reset(self, initial_records: Sequence[BeaconRecord]) -> None:
        self._fifo.clear()
        self._receiver_cache.clear()
        self._seq_counter = len(initial_records)
        snap = [
            BeaconRecord(
                sender_id=r.sender_id,
                timestamp=r.timestamp,
                seq_num=r.seq_num,
                pos=r.pos.copy(),
                vel=r.vel.copy(),
                energy_j=float(r.energy_j),
                cap_vec=r.cap_vec.copy(),
            )
            for r in initial_records
        ]
        # Pre-fill FIFO and receiver cache at t=0 so zero-order hold is well-defined from start
        max_len = max(1, self.cfg.beacon_delay_slots + self.slots_per_step + 1)
        for _ in range(max_len):
            self._fifo.append(snap)
        for i in range(self.num_uavs):
            for r in snap:
                if r.sender_id != i:
                    self._receiver_cache[(i, r.sender_id)] = r

    def push_and_receive(
        self,
        current_records: Sequence[BeaconRecord],
        rng: np.random.Generator,
        comm_noise_std: float = 0.0,
    ) -> List[List[BeaconRecord]]:
        """Push current step's beacons and return per-UAV received record lists."""
        snap: List[BeaconRecord] = []
        for r in current_records:
            self._seq_counter += 1
            snap.append(
                BeaconRecord(
                    sender_id=r.sender_id,
                    timestamp=r.timestamp,
                    seq_num=self._seq_counter,
                    pos=r.pos.copy(),
                    vel=r.vel.copy(),
                    energy_j=float(r.energy_j),
                    cap_vec=r.cap_vec.copy(),
                )
            )

        if self.cfg.beacon_delay_slots <= 0 and self.cfg.beacon_loss_prob <= 0.0 and comm_noise_std <= 0.0:
            return [snap for _ in range(self.num_uavs)]

        for _ in range(self.slots_per_step):
            self._fifo.append(snap)
        max_keep = self.cfg.beacon_delay_slots + self.slots_per_step + 2
        while len(self._fifo) > max_keep:
            self._fifo.popleft()

        idx = max(0, len(self._fifo) - 1 - self.cfg.beacon_delay_slots)
        delayed_batch = self._fifo[idx]

        per_receiver: List[List[BeaconRecord]] = []
        for recv_i in range(self.num_uavs):
            delivered_i: List[BeaconRecord] = []
            for rec in delayed_batch:
                if rec.sender_id == recv_i:
                    continue
                dropped = (
                    self.cfg.beacon_loss_prob > 0.0
                    and float(rng.random()) < self.cfg.beacon_loss_prob
                )
                if not dropped:
                    noisy_pos = rec.pos.copy()
                    noisy_vel = rec.vel.copy()
                    if comm_noise_std > 0.0:
                        noisy_pos = noisy_pos + rng.normal(0.0, comm_noise_std * 10.0, size=2)
                        noisy_vel = noisy_vel + rng.normal(0.0, comm_noise_std * 2.0, size=2)
                    updated = BeaconRecord(
                        sender_id=rec.sender_id,
                        timestamp=rec.timestamp,
                        seq_num=rec.seq_num,
                        pos=noisy_pos,
                        vel=noisy_vel,
                        energy_j=rec.energy_j,
                        cap_vec=rec.cap_vec.copy(),
                    )
                    self._receiver_cache[(recv_i, rec.sender_id)] = updated
                    delivered_i.append(updated)
                else:
                    # Zero-order hold on packet drop
                    cached = self._receiver_cache.get((recv_i, rec.sender_id))
                    if cached is not None:
                        delivered_i.append(cached)
            per_receiver.append(delivered_i)
        return per_receiver


def _segment_circle_intersects(
    p1: np.ndarray, p2: np.ndarray, center: np.ndarray, radius: float
) -> bool:
    """Check whether segment p1->p2 intersects a circle of given center and radius."""
    d = p2 - p1
    len_sq = float(np.dot(d, d))
    if len_sq <= 1e-12:
        return float(np.linalg.norm(p1 - center)) <= radius
    t = float(np.clip(np.dot(center - p1, d) / len_sq, 0.0, 1.0))
    proj = p1 + t * d
    return float(np.linalg.norm(proj - center)) <= radius


class StochasticWirelessChannel:
    """Stochastic LOS/NLOS/Rician channel & multi-source interference engine (Eqs. 8-12, S26)."""

    def __init__(self, cfg: WirelessConfig, workspace_size: float = 100.0) -> None:
        self.cfg = cfg
        self.workspace_size = workspace_size
        self.noise_power_w: float = cfg.n0 * cfg.bandwidth  # 4e-15 W
        self.uav_los_matrix: np.ndarray | None = None       # shape (N, N) bool, fixed per episode
        self.ground_los_matrix: np.ndarray | None = None    # shape (Q, N) bool, fixed per episode
        self.ground_positions: np.ndarray = np.zeros((0, 2), dtype=np.float64)

    def reset_episode(
        self,
        uav_positions: np.ndarray,
        obstacle_positions: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Sample episode-level LOS/NLOS states and ground interferer coordinates (Table S26)."""
        n_uav = uav_positions.shape[0]
        # Sample 6 ground interferers uniformly in annulus [30, 60] m around (50, 50)
        q_cnt = self.cfg.num_ground_interferers
        if q_cnt > 0:
            center = np.array([0.5 * self.workspace_size, 0.5 * self.workspace_size], dtype=np.float64)
            u_r = rng.uniform(
                self.cfg.ground_annulus_r_in ** 2,
                self.cfg.ground_annulus_r_out ** 2,
                size=q_cnt,
            )
            radii = np.sqrt(u_r)
            angles = rng.uniform(0.0, 2.0 * np.pi, size=q_cnt)
            self.ground_positions = center[None, :] + np.stack(
                [radii * np.cos(angles), radii * np.sin(angles)], axis=-1
            )
        else:
            self.ground_positions = np.zeros((0, 2), dtype=np.float64)

        if self.cfg.wireless_mode.lower() == "lf":
            self.uav_los_matrix = np.ones((n_uav, n_uav), dtype=bool)
            self.ground_los_matrix = np.ones((q_cnt, n_uav), dtype=bool)
        else:
            # E1: Stochastic LOS/NLOS fixed per episode, incorporating initial obstacle blockage
            los_mat = np.ones((n_uav, n_uav), dtype=bool)
            for i in range(n_uav):
                for j in range(i + 1, n_uav):
                    blocked = any(
                        _segment_circle_intersects(uav_positions[i], uav_positions[j], obs_c, 1.5)
                        for obs_c in obstacle_positions
                    )
                    p_nlos = min(0.85, self.cfg.nlos_prob + (0.35 if blocked else 0.0))
                    is_los = float(rng.random()) >= p_nlos
                    los_mat[i, j] = is_los
                    los_mat[j, i] = is_los
            self.uav_los_matrix = los_mat

            g_los = rng.random((q_cnt, n_uav)) >= self.cfg.nlos_prob
            self.ground_los_matrix = g_los

    def sample_rician_power_fading(
        self, is_los: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample unit-mean Rician power fading |h|^2 with E[|h|^2] = 1 (Table S26)."""
        k_factor = np.where(is_los, self.cfg.k_los, self.cfg.k_nlos)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=is_los.shape)
        x = rng.normal(0.0, 1.0, size=is_los.shape)
        y = rng.normal(0.0, 1.0, size=is_los.shape)
        los_amp = np.sqrt(k_factor / (k_factor + 1.0))
        scat_amp = np.sqrt(1.0 / (2.0 * (k_factor + 1.0)))
        h_real = los_amp * np.cos(phi) + scat_amp * x
        h_imag = los_amp * np.sin(phi) + scat_amp * y
        return h_real * h_real + h_imag * h_imag

    def compute_channel_gain_matrix(
        self,
        tx_positions: np.ndarray,
        rx_positions: np.ndarray,
        los_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Compute G_ab^{ch,t} = G_a * G_b * (d0 / (d + d0))^{n_s} * |h|^2 (Eq. 12)."""
        diff = tx_positions[:, None, :] - rx_positions[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        exponents = np.where(los_mask, self.cfg.n_los, self.cfg.n_nlos)
        path_loss = np.power(self.cfg.d0 / (dists + self.cfg.d0), exponents)
        fading = self.sample_rician_power_fading(los_mask, rng)
        g_ant = self.cfg.antenna_gain * self.cfg.antenna_gain
        return g_ant * path_loss * fading

    def evaluate_step_links(
        self,
        uav_positions: np.ndarray,
        power_matrix: np.ndarray,
        rng: np.random.Generator,
        ground_power_override: float | None = None,
    ) -> Dict[str, np.ndarray]:
        """Evaluate SINR (Eq. 9), UAV/ground interference (Eqs. 10-11), and effective links (Eq. 8).

        Args:
            uav_positions: shape (N, 2) [m]
            power_matrix: shape (N, N) [W], directed transmit power P_{ij}^t (0 on non-candidates & diag)
            rng: numpy Generator
            ground_power_override: optional per-source ground interference power [W]

        Returns dictionary containing:
            - 'g_uav': shape (N, N) channel gains
            - 'sinr_linear': shape (N, N) linear SINR
            - 'sinr_db': shape (N, N) 10*log10(max(sinr_linear, 1e-30))
            - 'attempted': shape (N, N) bool (P_{ij} > P_min, off-diagonal)
            - 'effective': shape (N, N) bool (attempted & SINR >= SINR_th)
            - 'uav_interference': shape (N, N) I_{ij, U}^t [W]
            - 'ground_interference': shape (N,) I_{j, G}^t [W]
        """
        n_uav = uav_positions.shape[0]
        if self.uav_los_matrix is None or self.uav_los_matrix.shape[0] != n_uav:
            self.reset_episode(uav_positions, np.zeros((0, 2)), rng)
        assert self.uav_los_matrix is not None

        p_mat = np.asarray(power_matrix, dtype=np.float64).copy()
        np.fill_diagonal(p_mat, 0.0)

        g_uav = self.compute_channel_gain_matrix(
            uav_positions, uav_positions, self.uav_los_matrix, rng
        )
        np.fill_diagonal(g_uav, 0.0)

        # Eq. (10): For link i -> j, interferers are l not in {i, j} with total tx power P_{l, tx}^t
        p_tx_total = np.sum(p_mat, axis=1)  # shape (N,)
        # Total received power from all l != j at receiver j: sum_{l != j} P_{l,tx} * G_{l,j}
        total_rx_from_uavs_at_j = p_tx_total @ g_uav  # shape (N,)
        # Subtract sender i's own total power contribution P_{i,tx} * G_{i,j} to exclude l == i
        i_uav = total_rx_from_uavs_at_j[None, :] - (p_tx_total[:, None] * g_uav)
        i_uav = np.maximum(0.0, i_uav)
        np.fill_diagonal(i_uav, 0.0)

        # Eq. (11): Ground interference I_{j, G}^t
        p_ground = (
            self.cfg.ground_power_w if ground_power_override is None else float(ground_power_override)
        )
        q_cnt = self.ground_positions.shape[0]
        if p_ground > 0.0 and q_cnt > 0 and self.ground_los_matrix is not None:
            active_q = (rng.random(q_cnt) < self.cfg.ground_active_prob).astype(np.float64)
            g_ground = self.compute_channel_gain_matrix(
                self.ground_positions, uav_positions, self.ground_los_matrix, rng
            )  # shape (Q, N)
            i_ground = (active_q * p_ground) @ g_ground  # shape (N,)
        else:
            i_ground = np.zeros(n_uav, dtype=np.float64)

        # Eq. (9): Exact linear SINR (no extra stabilizer because N0*B > 0)
        denom = self.noise_power_w + i_uav + i_ground[None, :]
        sinr_linear = (p_mat * g_uav) / denom
        np.fill_diagonal(sinr_linear, 0.0)

        attempted = p_mat > self.cfg.p_min
        np.fill_diagonal(attempted, False)
        effective = attempted & (sinr_linear >= self.cfg.sinr_th)
        sinr_db = 10.0 * np.log10(np.maximum(sinr_linear, 1e-30))

        return {
            "g_uav": g_uav,
            "sinr_linear": sinr_linear,
            "sinr_db": sinr_db,
            "attempted": attempted,
            "effective": effective,
            "uav_interference": i_uav,
            "ground_interference": i_ground,
        }
