from __future__ import annotations

import numpy as np
import pytest

from configs.defaults import (
    EnvConfig,
    S29_DEMAND_CONDITIONS,
    WirelessConfig,
    project_capped_simplex,
)
from envs.lf_uav_env import LowFidelityUAVEnv
from envs.service_and_reward import (
    compute_routing_weights,
    compute_service_quality,
    compute_specialization_index,
    evaluate_s19_counterfactual,
)
from envs.wireless import (
    BeaconChannelBuffer,
    BeaconRecord,
    StochasticWirelessChannel,
    deduplicate_and_select_candidates,
)


def test_candidate_dedup_tie_break_and_truncation_s16() -> None:
    """Verify Eq. (7), Alg. S1 Step 1, and Table S16 self-exclusion, timestamp/seq dedup & M_cap=3."""
    self_pos = np.array([50.0, 50.0])
    records = [
        # Self record -> must be excluded
        BeaconRecord(0, timestamp=1.2, seq_num=99, pos=np.array([50.0, 50.0]), vel=np.zeros(2), energy_j=90000.0, cap_vec=np.ones(6)),
        # Duplicate UAV 1: older timestamp vs newer timestamp
        BeaconRecord(1, timestamp=1.0, seq_num=10, pos=np.array([51.0, 50.0]), vel=np.zeros(2), energy_j=80000.0, cap_vec=np.ones(6)),
        BeaconRecord(1, timestamp=1.2, seq_num=11, pos=np.array([60.0, 50.0]), vel=np.zeros(2), energy_j=75000.0, cap_vec=np.ones(6)),
        # Duplicate UAV 2: equal timestamp -> broken by larger seq_num
        BeaconRecord(2, timestamp=1.2, seq_num=12, pos=np.array([99.0, 99.0]), vel=np.zeros(2), energy_j=80000.0, cap_vec=np.ones(6)),
        BeaconRecord(2, timestamp=1.2, seq_num=15, pos=np.array([55.0, 50.0]), vel=np.zeros(2), energy_j=82000.0, cap_vec=np.ones(6)),
        # UAV 3 within range (dist 20 m)
        BeaconRecord(3, timestamp=1.2, seq_num=16, pos=np.array([70.0, 50.0]), vel=np.zeros(2), energy_j=90000.0, cap_vec=np.ones(6)),
        # UAV 4 within range (dist 30 m) -> 4th unique neighbor, truncated by M_cap=3
        BeaconRecord(4, timestamp=1.2, seq_num=17, pos=np.array([80.0, 50.0]), vel=np.zeros(2), energy_j=95000.0, cap_vec=np.ones(6)),
        # UAV 5 outside d_comm=50 m (dist 55 m)
        BeaconRecord(5, timestamp=1.2, seq_num=18, pos=np.array([105.0, 50.0]), vel=np.zeros(2), energy_j=95000.0, cap_vec=np.ones(6)),
    ]
    res = deduplicate_and_select_candidates(0, self_pos, records, d_comm=50.0, m_cap=3)
    assert res.unique_count == 4
    # Nearest 3 must be UAV 2 (5 m), UAV 1 (10 m from newer timestamp!), UAV 3 (20 m)
    assert res.valid_ids == [2, 1, 3]
    assert np.all(res.valid_mask == np.array([True, True, True]))
    assert np.allclose(res.padded_dist, [5.0, 10.0, 20.0])

    # Verify Table S16 exact arithmetic mean: (1*1000 + 2*1100 + 3*800 + 3*7100)/10000 == 2.69
    counts = [min(u, 3) for u in ([1] * 1000 + [2] * 1100 + [3] * 800 + [4] * 7100)]
    assert len(counts) == 10_000
    assert abs(float(np.mean(counts)) - 2.69) < 1e-12


def test_rician_unit_mean_and_sinr_interference() -> None:
    """Verify unit-mean Rician fading E[|h|^2]=1 and SINR / interference equations (Eqs. 8-12)."""
    w_cfg = WirelessConfig(wireless_mode="e1", ground_power_w=2.0)
    ch = StochasticWirelessChannel(w_cfg)
    rng = np.random.Generator(np.random.PCG64(42))

    los_samples = ch.sample_rician_power_fading(np.ones(50_000, dtype=bool), rng)
    nlos_samples = ch.sample_rician_power_fading(np.zeros(50_000, dtype=bool), rng)
    assert abs(float(np.mean(los_samples)) - 1.0) < 0.02
    assert abs(float(np.mean(nlos_samples)) - 1.0) < 0.02

    pos = np.array([[20.0, 20.0], [30.0, 20.0], [25.0, 25.0], [60.0, 60.0], [70.0, 70.0]])
    ch.reset_episode(pos, np.array([[45.0, 45.0]]), rng)
    # Check annulus [30, 60] m around (50, 50)
    radii = np.linalg.norm(ch.ground_positions - np.array([[50.0, 50.0]]), axis=-1)
    assert np.all((radii >= 30.0 - 1e-9) & (radii <= 60.0 + 1e-9))

    p_mat = np.zeros((5, 5), dtype=np.float64)
    p_mat[0, 1] = 0.5   # > P_min=0.02 -> attempted
    p_mat[1, 0] = 0.01  # <= P_min=0.02 -> NOT attempted
    out = ch.evaluate_step_links(pos, p_mat, rng)
    assert bool(out["attempted"][0, 1]) is True
    assert bool(out["attempted"][1, 0]) is False
    assert np.all(np.diag(out["sinr_linear"]) == 0.0)


def test_beacon_delay_and_loss_fifo() -> None:
    """Verify E3 200-ms FIFO beacon delay and packet-loss zero-order hold (Table S26)."""
    w_cfg = WirelessConfig(beacon_delay_slots=5, beacon_loss_prob=1.0)  # 100% drop -> holds t=0 cache
    buf = BeaconChannelBuffer(num_uavs=2, cfg=w_cfg)
    rng = np.random.Generator(np.random.PCG64(7))
    init_recs = [
        BeaconRecord(0, 0.0, 0, np.array([10.0, 10.0]), np.zeros(2), 90000.0, np.ones(6)),
        BeaconRecord(1, 0.0, 1, np.array([20.0, 20.0]), np.zeros(2), 90000.0, np.ones(6)),
    ]
    buf.reset(init_recs)
    moved_recs = [
        BeaconRecord(0, 0.2, 2, np.array([35.0, 35.0]), np.zeros(2), 89000.0, np.ones(6)),
        BeaconRecord(1, 0.2, 3, np.array([45.0, 45.0]), np.zeros(2), 89000.0, np.ones(6)),
    ]
    delivered = buf.push_and_receive(moved_recs, rng)
    # With 100% loss, UAV 0 must still see UAV 1's cached initial position [20, 20]
    assert len(delivered[0]) == 1
    assert np.allclose(delivered[0][0].pos, [20.0, 20.0])


def test_service_rules_add_geo_harm_boundaries() -> None:
    """Verify Eqs. (16)-(19) ADD >= GEO >= HARM, zero-sum, zero-weight omission, and zero-cap exact 0."""
    pos = np.array([[10.0, 10.0], [10.0, 10.0]])
    caps = np.array(
        [
            [1.0, 100.0, 6.0, 2.8, 1.2, 1.1],
            [1.0, 100.0, 6.0, 2.8, 1.2, 1.1],
        ]
    )
    # UAV 0 has positive energy & chi; UAV 1 has chi = 0 (zero comm capability on positive weight)
    energy = np.array([120_000.0, 120_000.0])
    chi = np.array([0.8, 0.0])
    task_centers = np.array([[15.0, 10.0], [20.0, 10.0], [25.0, 10.0]])
    demands = np.array(
        [
            [1.0, 0.6, 0.8, 0.5, 0.4],  # Task 0: all positive weights
            [1.0, 0.7, 0.5, 0.0, 0.0],  # Task 1: zero weights on energy & comm!
            [1.0, 0.0, 0.0, 0.0, 0.0],  # Task 2: s_k == 0 -> all rules must be 0
        ]
    )

    q_add = compute_service_quality(pos, caps, energy, chi, task_centers, demands, "ADD")
    q_geo = compute_service_quality(pos, caps, energy, chi, task_centers, demands, "GEO")
    q_harm = compute_service_quality(pos, caps, energy, chi, task_centers, demands, "HARM")

    assert np.all(np.isfinite(q_add)) and np.all(np.isfinite(q_geo)) and np.all(np.isfinite(q_harm))
    # Weighted AM >= GM >= HM on strictly positive components (UAV 0, Task 0)
    assert q_add[0, 0] >= q_geo[0, 0] - 1e-12 >= q_harm[0, 0] - 2e-12 > 0.0
    # UAV 1 has chi=0 on Task 0 (where w_comm=0.4 > 0): GEO and HARM must be exact 0.0!
    assert q_geo[1, 0] == 0.0
    assert q_harm[1, 0] == 0.0
    assert q_add[1, 0] > 0.0
    # On Task 1, w_comm == 0 is omitted -> UAV 1 with chi=0 MUST still get positive GEO & HARM!
    assert q_geo[1, 1] > 0.0 and q_harm[1, 1] > 0.0
    # On Task 2, s_k == 0 -> all three rules equal 0.0
    assert np.allclose(q_add[:, 2], 0.0)
    assert np.allclose(q_geo[:, 2], 0.0)
    assert np.allclose(q_harm[:, 2], 0.0)


def test_s29_capped_simplex_and_specialization_and_s19() -> None:
    """Verify Table S29 exact demand entries, Eqs. (41)-(42) S_spec, S9 layouts, and S19."""
    # Check printed Table S29 entries (e.g. D2 Task 2: [1, 1.0, 0.616667, 0.366667, 0.316667],
    # D6 Task 2: [1, 1.0, 0.469841, 0.469841, 0.360317])
    d2_t2 = S29_DEMAND_CONDITIONS["D2"][1]
    assert np.allclose(d2_t2, [1.0, 1.0, 0.6166667, 0.3666667, 0.3166667], atol=1e-6)
    d6_t2 = S29_DEMAND_CONDITIONS["D6"][1]
    assert np.allclose(d6_t2, [1.0, 1.0, 0.4698413, 0.4698413, 0.3603175], atol=1e-6)

    w_proj = project_capped_simplex(np.array([1.8, -0.2, 0.5, 0.4]), total_sum=2.1)
    assert np.all((w_proj >= 0.0) & (w_proj <= 1.0))
    assert abs(float(np.sum(w_proj)) - 2.1) < 1e-9

    # S_spec: zero-service fallback gives uniform shares -> S_spec near 0; single-UAV dominance -> near 1
    zero_svc = np.zeros((5, 3))
    assert abs(compute_specialization_index(zero_svc)) < 1e-6
    one_hot_svc = np.zeros((5, 3))
    one_hot_svc[0, 0] = 10.0
    one_hot_svc[1, 1] = 10.0
    one_hot_svc[2, 2] = 10.0
    assert compute_specialization_index(one_hot_svc) > 0.99

    # S9 'more_obstructed' must generate 8 obstacles (5 parent + 3 corridor obstacles)
    env_obs8 = LowFidelityUAVEnv(EnvConfig(scene_layout="more_obstructed"), training=False)
    env_obs8.reset(seed=320_000)
    assert env_obs8.obstacle_centers.shape == (8, 2)

    # Alternative 20-m routing check
    _, alpha_alt, _ = compute_routing_weights(
        np.array([[10.0, 10.0], [80.0, 80.0]]),
        np.array([[15.0, 10.0], [22.0, 10.0], [90.0, 90.0]]),
        np.array([0.0, 0.0, 45.0]),  # Task 2 already completed -> u_2 = 0
        np.array([45.0, 45.0, 45.0]),
        routing_mode="alt_20m",
    )
    assert np.allclose(alpha_alt[0], [0.5, 0.5, 0.0])
    assert np.allclose(alpha_alt[1], [0.0, 0.0, 0.0])  # Only finished Task 2 is within 20 m!

    s19 = evaluate_s19_counterfactual()
    assert s19["n_states"] == 512 and s19["n_entries"] == 1341
    assert 0.70 < s19["spearman_v1"] <= 1.0
