from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


# ============================================================================
# Manuscript Table 1: Five heterogeneous UAV platform classes
# Order of 6D capability descriptor z_i^LF (Eq. 3):
#   [m (kg), E_max (kJ), v_max (m/s), a_max (m/s^2), xi_sen, g]
# ============================================================================
PLATFORM_TYPES: Tuple[str, ...] = ("fast", "sensing", "relay", "endurance", "general")

PLATFORM_SPECS: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "fast":      (0.90,  90.0, 8.0, 3.5, 0.8, 0.9),
    "sensing":   (1.20, 105.0, 5.5, 2.5, 1.5, 0.9),
    "relay":     (1.10, 110.0, 5.8, 2.6, 0.9, 1.5),
    "endurance": (1.35, 150.0, 5.0, 2.2, 0.9, 1.1),
    "general":   (1.00, 100.0, 6.0, 2.8, 1.0, 1.0),
}

# Supplementary Table S25 (E-HET): Aggregate-matched homogeneous UAV specification
# Matches 5-UAV heterogeneous total mass (5.55 kg), energy (555 kJ), and gain (5.4)
HOMOGENEOUS_SPEC: Tuple[float, float, float, float, float, float] = (
    1.11, 111.0, 6.0, 2.8, 1.0, 1.08
)

# Supplementary Table S13 (E5): Held-out platform capability descriptor shifts
# Format: condition_name -> (replaced_slot_type, new_6d_descriptor)
S13_DESCRIPTOR_SHIFTS: Dict[str, Tuple[str, Tuple[float, float, float, float, float, float]]] = {
    # Three convex interpolations
    "fast_general_interp": (
        "fast",
        (0.935, 93.5, 7.3, 3.255, 0.87, 0.935),       # 0.65 * z_F + 0.35 * z_G
    ),
    "sensing_general_interp": (
        "sensing",
        (1.08, 102.0, 5.8, 2.68, 1.20, 0.96),         # 0.40 * z_S + 0.60 * z_G
    ),
    "relay_endurance_interp": (
        "relay",
        (1.2125, 128.0, 5.44, 2.42, 0.90, 1.32),      # 0.55 * z_R + 0.45 * z_E
    ),
    # Four coordinate-range extrapolations
    "high_sensing_low_endurance": (
        "sensing",
        (1.16, 78.0, 5.3, 2.4, 1.68, 0.94),
    ),
    "low_speed_high_gain": (
        "relay",
        (1.13, 116.0, 4.4, 2.1, 0.88, 1.72),
    ),
    "high_energy_high_mass": (
        "endurance",
        (1.48, 168.0, 4.8, 2.05, 0.92, 1.12),
    ),
    "fast_weak_communication": (
        "fast",
        (0.86, 86.0, 8.6, 3.65, 0.76, 0.72),
    ),
}

# Supplementary Table S26: Fixed max-scale normalization constants for z_i
# [m_ref^norm (kg), E_ref^max (kJ), v_max^ref (m/s), a_ref^max (m/s^2), xi_ref^sen, g_ref]
# Extrapolation values > 1.0 are NOT clipped.
CAP_NORM_REF: Tuple[float, float, float, float, float, float] = (
    1.35, 150.0, 8.0, 3.5, 1.5, 1.5
)


# ============================================================================
# Manuscript Table 2: Task-demand profiles
# Order of 5D demand vector d_k^t (Eq. 4):
#   [rho (intensity), nu (mobility), sigma (sensing), varepsilon (endurance), lambda (comm)]
# ============================================================================
TASK_TYPES: Tuple[str, ...] = ("Reconnaissance", "Sensing", "Relay")

NOMINAL_TASK_PROFILES: Dict[str, Tuple[float, float, float, float, float]] = {
    "Reconnaissance": (1.0, 1.0, 0.4, 0.4, 0.3),
    "Sensing":        (1.0, 0.4, 1.0, 0.5, 0.4),
    "Relay":          (1.0, 0.3, 0.3, 0.8, 1.0),
}

# Mean neutral Query vector used in QUERY control (Supplementary p. 8, Table 8)
NEUTRAL_QUERY_DEMAND: Tuple[float, float, float, float, float] = (
    1.0, 1.7 / 3.0, 1.7 / 3.0, 1.7 / 3.0, 1.7 / 3.0
)

# Supplementary Table S4: Held-out deterministic demand combinations
HELD_OUT_DEMAND_PROFILES: Dict[str, Tuple[float, float, float, float, float]] = {
    "mobility_communication": (1.0, 0.9, 0.3, 0.4, 0.9),
    "sensing_endurance":      (1.0, 0.3, 0.9, 0.9, 0.4),
    "balanced_high":          (1.0, 0.8, 0.8, 0.8, 0.8),
}


def project_capped_simplex(
    v: np.ndarray,
    total_sum: float,
    lower: float = 0.0,
    upper: float = 1.0,
    tol: float = 1e-11,
    max_iter: int = 100,
) -> np.ndarray:
    """Project vector v onto {w : lower <= w_i <= upper, sum(w) = total_sum} in L2 norm.

    Implements Supplementary Table S29 exact capped-simplex projection:
        arg min_{0 <= w <= 1, sum(w) = s_k} ||w - v||_2^2
    via bisection on the Lagrange multiplier mu where w_i(mu) = clip(v_i - mu, lower, upper).
    """
    v_arr = np.asarray(v, dtype=np.float64)
    n = v_arr.size
    if total_sum <= n * lower + tol:
        return np.full(n, lower, dtype=np.float64)
    if total_sum >= n * upper - tol:
        return np.full(n, upper, dtype=np.float64)

    lo = float(np.min(v_arr) - upper - 1.0)
    hi = float(np.max(v_arr) - lower + 1.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        w = np.clip(v_arr - mid, lower, upper)
        s = float(np.sum(w))
        if abs(s - total_sum) <= tol:
            return w
        if s > total_sum:
            lo = mid
        else:
            hi = mid
    return np.clip(v_arr - 0.5 * (lo + hi), lower, upper)


def build_s29_demand_profiles() -> Dict[str, List[np.ndarray]]:
    """Construct all 7 demand conditions D0--D6 defined in Supplementary Table S29.

    Returns a dictionary mapping 'D0'..'D6' to a list of three 5D vectors [rho=1, nu, sigma, eps, lam].
    """
    base_w = [
        np.array([1.0, 0.4, 0.4, 0.3], dtype=np.float64),  # Task 1: s_1 = 2.1
        np.array([0.4, 1.0, 0.5, 0.4], dtype=np.float64),  # Task 2: s_2 = 2.3
        np.array([0.3, 0.3, 0.8, 1.0], dtype=np.float64),  # Task 3: s_3 = 2.4
    ]
    sums = [float(np.sum(w)) for w in base_w]

    out: Dict[str, List[np.ndarray]] = {}
    # D0: nominal
    out["D0"] = [np.concatenate(([1.0], w.copy())) for w in base_w]
    # D1: equalized s_k / 4
    out["D1"] = [np.concatenate(([1.0], np.full(4, s / 4.0))) for s in sums]
    # D2..D5: single-attribute emphasis on dim d in {0, 1, 2, 3}
    for d_idx, cond in enumerate(("D2", "D3", "D4", "D5")):
        e_d = np.zeros(4, dtype=np.float64)
        e_d[d_idx] = 1.0
        cond_list: List[np.ndarray] = []
        for w0, s_k in zip(base_w, sums):
            v = 0.5 * w0 + 0.5 * s_k * e_d
            w_proj = project_capped_simplex(v, s_k, 0.0, 1.0)
            cond_list.append(np.concatenate(([1.0], w_proj)))
        out[cond] = cond_list
    # D6: rotate task vectors (3 -> 1, 1 -> 2, 2 -> 3) and rescale to destination s_k, then project
    rotated_src = [base_w[2], base_w[0], base_w[1]]
    d6_list: List[np.ndarray] = []
    for src_w, s_dest in zip(rotated_src, sums):
        v = src_w * (s_dest / float(np.sum(src_w)))
        w_proj = project_capped_simplex(v, s_dest, 0.0, 1.0)
        d6_list.append(np.concatenate(([1.0], w_proj)))
    out["D6"] = d6_list
    return out


S29_DEMAND_CONDITIONS: Dict[str, List[np.ndarray]] = build_s29_demand_profiles()


# ============================================================================
# Configuration Dataclasses (Tables 3, S2, S17, S22, S26, S31, S32, S48, S53-S56)
# ============================================================================
@dataclass
class WirelessConfig:
    """Wireless channel, interference, and observation-kernel parameters (Eqs. 7-12, Table S26)."""
    d_comm: float = 50.0                  # Communication discovery radius [m]
    m_cap: int = 3                        # Max retained nearest neighbors M_cap
    p_max: float = 1.0                    # Per-UAV max transmit power budget [W]
    p_min: float = 0.02                   # Minimum attempted-link power threshold [W]
    sinr_th: float = 1.0                  # Effective-link linear SINR threshold (0 dB)
    sinr_ref: float = 1.0                 # History-normalization linear SINR reference (0 dB)
    n0: float = 4e-21                     # Noise spectral density [W/Hz] (-174 dBm/Hz)
    bandwidth: float = 1e6                # Channel bandwidth B [Hz] -> N0*B = 4e-15 W
    d0: float = 1.0                       # Path-loss reference distance [m]
    n_los: float = 2.2                    # LOS path-loss exponent
    n_nlos: float = 3.8                   # NLOS path-loss exponent
    k_los: float = 10.0                   # LOS Rician K-factor (linear)
    k_nlos: float = 2.0                   # NLOS Rician K-factor (linear)
    antenna_gain: float = 1.0             # Omnidirectional antenna gain G_a = G_b = 1.0
    wireless_mode: str = "e1"             # 'lf' (LOS-dominant) or 'e1' (stochastic LOS/NLOS/Rician)
    nlos_prob: float = 0.35               # Base NLOS probability in stochastic E1 channel
    # Ground interference (E4, Table S26)
    num_ground_interferers: int = 6
    ground_power_w: float = 0.0           # 0.0 nominal; 1.0 or 2.0 W in E4 / INT sweeps
    ground_active_prob: float = 0.5       # Bernoulli(0.5) per source per step
    ground_annulus_r_in: float = 30.0     # Inner radius around workspace center (50, 50) [m]
    ground_annulus_r_out: float = 60.0    # Outer radius around workspace center (50, 50) [m]
    # Beacon delay / loss observation perturbations (E3, Table S26)
    beacon_slot_ms: float = 40.0          # 40 ms beacon slot (5 slots = 200 ms policy step)
    beacon_delay_slots: int = 0           # 0 in training; e.g. 5 (200 ms) in E3
    beacon_loss_prob: float = 0.0         # 0.0 in training; e.g. 0.5 in E3
    eps_comm: float = 1e-6                # Stabilizer in chi_i and Comm success


@dataclass
class EnvConfig:
    """Low-fidelity 2D heterogeneous UAV Dec-POMDP configuration (Tables 1-3, S26)."""
    workspace_size: float = 100.0         # [0, 100] x [0, 100] m
    num_uavs: int = 5                     # Team size N (supports 5, 10, 20, ...)
    num_tasks: int = 3                    # Number of task regions K
    num_obstacles: int = 5                # Number of circular obstacles N_obs (8 in S9 obstructed)
    obs_cap: int = 5                      # Max nearest obstacles encoded by Actor f_obs
    max_steps: int = 1500                 # Horizon T = 1500 steps (300 s)
    dt: float = 0.2                       # Policy decision step duration [s]
    # Service & routing (Eqs. 13-19, 23, Table S26)
    service_rule: str = "ADD"             # 'ADD', 'GEO', 'HARM' (Eqs. 17-19), or 'E7_MIN'/'E7_PROD'
    routing_mode: str = "softmax"         # 'softmax' (Eq. 23) or 'alt_20m' (S28 ROUTE control)
    beta_d: float = 0.05                  # Distance decay in service quality [m^-1]
    gamma_d: float = 2.0                  # Distance weight in routing score Eq. (23)
    gamma_u: float = 1.0                  # Unfinished-progress weight in routing score Eq. (23)
    d_norm: float = 100.0                 # Distance normalization scale [m]
    r_service: float = 20.0               # Sensing-power activation & alt-routing radius [m]
    task_req_progress: float = 45.0       # Required progress threshold Gamma_k^req
    train_demand_noise_std: float = 0.03  # Small clipped Gaussian demand jitter during training
    # Energy parameters (SI W / J, Eq. 20, Table S26)
    c_prop_base: float = 25.0             # Baseline propulsion power [W]
    c_prop_v: float = 1.0                 # Speed-squared propulsion coeff [W s^2 / m^2]
    m_ref_energy: float = 1.0             # Reference mass for propulsion power [kg]
    c_sen: float = 5.0                    # Sensing power coeff [W]
    c_comp: float = 5.0                   # Onboard computation power [W]
    v_max_ref: float = 8.0                # Speed normalization reference [m/s]
    e_max_ref_j: float = 150000.0         # Energy normalization reference (150 kJ = 150,000 J)
    # Safety & collision thresholds (Table 3)
    d_col_uav: float = 0.4                # UAV-UAV center-distance collision threshold [m]
    d_col_obs: float = 0.5                # UAV-obstacle center-distance collision threshold [m]
    d_nearmiss_margin: float = 1.0        # Surface-gap near-miss margin [m]
    # Reward weights (Eq. 21, Table S26)
    w1_task: float = 1.0
    w2_energy: float = 0.1
    w3_collision: float = 0.5
    w4_comm: float = 0.2
    w5_constraint: float = 0.3
    # Scenario / heterogeneity overrides
    homogeneous_team: bool = False        # E-HET homogeneous 5-UAV baseline
    s13_shift: str | None = None          # Optional S13 descriptor shift key
    scene_layout: str = "nominal"         # 'nominal', 'moderately_shifted', 'more_obstructed' (S9)
    demand_mode: str = "nominal"          # 'nominal', S4 key, 'continuous_random', or 'D0'..'D6'
    eps_log: float = 1e-8                 # Log safeguard (Eq. 30, 42)
    wireless: WirelessConfig = field(default_factory=WirelessConfig)


@dataclass
class ModelConfig:
    """Architecture, attention, and prior-guided action configuration (Eqs. 22-37, Table 3, S7)."""
    method: str = "mtc_mappo"
    # Supported methods:
    #   'mtc_mappo', 'morph_gat', 'gat_mappo', 'mappo_morph', 'role_mappo', 'mappo', 'tarmac_b1'
    # Ablation flags for MTC-MAPPO (Table 5 / E2 & Table 7):
    ablate_capability_descriptor: bool = False
    ablate_task_attention: bool = False
    ablate_link_energy_mod: bool = False
    ablate_comm_prior: bool = False
    # Core dimensions (Table 3)
    embed_dim: int = 64
    num_heads: int = 2
    actor_hidden: int = 128
    critic_hidden: int = 256
    m_cap: int = 3
    obs_cap: int = 5
    num_tasks: int = 3
    # Attention & matching-score hyperparameters (Eq. 26, Table S48)
    tau_a: float = 0.5                    # Cosine-similarity temperature
    alpha_d: float = 0.5                  # Distance penalty weight
    alpha_c: float = 0.4                  # Link-history penalty weight
    alpha_e: float = 0.3                  # Neighbor energy-deficiency penalty weight
    gamma_d: float = 2.0                  # Routing distance weight
    gamma_u: float = 1.0                  # Routing unfinished-fraction weight
    d_norm: float = 100.0                 # Distance scale [m]
    sinr_ref: float = 1.0                 # Linear SINR reference
    # Prior & Gaussian clamping (Eqs. 29-35)
    eps_s: float = 0.01                   # Prior smoothing constant
    eps_log: float = 1e-8                 # Log-prior safeguard
    log_std_min: float = -4.0             # Diagonal Gaussian min log-std
    log_std_max: float = 0.5              # Diagonal Gaussian max log-std
    p_max: float = 1.0                    # Per-UAV transmit power budget [W]
    # Inference-time intervention hooks
    neutral_query_override: bool = False  # QUERY control (Table 8)
    prior_uniform_mix: float = 0.0        # 0.0 original; 0.2 in SMOOTH/S40-S43; 1.0 uniform


@dataclass
class PPOConfig:
    """CTDE-MAPPO training hyperparameters (Table 3 & Supplementary Table S31)."""
    total_env_steps: int = 1_300_000      # 1.3e6 joint environment steps
    rollout_steps: int = 4096             # Environment steps per full rollout
    minibatch_size: int = 512             # Max valid entries per minibatch
    ppo_epochs: int = 5                   # Shuffled epochs per rollout
    gamma: float = 0.99                   # Discount factor
    gae_lambda: float = 0.95              # GAE recursion parameter
    clip_eps: float = 0.20                # PPO ratio & value-loss clip half-width
    actor_lr: float = 3e-4                # Adam learning rate (actor + encoders + attn)
    critic_lr: float = 3e-4               # Adam learning rate (centralized critic)
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-5
    weight_decay: float = 0.0
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5            # Separate global grad-norm clip for actor & critic
    adv_norm_eps: float = 1e-8            # Population-SD denominator stabilizer


@dataclass
class CompensatorConfig:
    """Execution-layer disturbance predictor & 1st-order residual estimator (Tables S53-S56)."""
    group: str = "A"                      # 'A' (PID), 'B' (PID + pred FF), 'C' (PID + pred FF + residual)
    dt_ctrl: float = 0.01                 # 100 Hz control step [s]
    tau_f: float = 0.30                   # First-order predictor time constant [s]
    k_e: float = 0.08                     # Residual estimator gain K_e
    a_max_ff: float = 2.0                 # Feedforward / residual clipping bound [m/s^2]
    # Disturbance & prediction-error regime ('nominal', 'low_error', 'high_error', 'step_1p5')
    disturbance_mode: str = "nominal"
    ou_sigma_d: float = 0.6               # Stationary per-axis OU disturbance SD [m/s^2]
    ou_tau_c: float = 2.0                 # OU correlation time [s]
    high_error_noise_std: float = 0.7     # Injected per-axis predictor Gaussian noise SD [m/s^2]
    # Single-platform step disturbance (Table S53)
    step_accel_mps2: float = 1.5          # Fixed-direction horizontal disturbance [m/s^2]
    step_t_start: float = 30.0            # Disturbance onset [s]
    step_t_end: float = 90.0              # Disturbance removal [s]
    step_t_total: float = 150.0           # Total observation window [s]
    recovery_tol_mps: float = 0.10        # Velocity-error tolerance [m/s]
    recovery_hold_s: float = 5.0          # Continuous confirmation duration [s]


@dataclass
class PyBulletConfig:
    """Six-degree-of-freedom PyBullet quadrotor & 100 Hz cascaded PID settings (Tables S22, S26)."""
    gui: bool = False                     # False -> pybullet.DIRECT headless mode
    ctrl_freq_hz: int = 100               # 100 Hz low-level flight control
    policy_freq_hz: int = 5               # 5 Hz high-level policy decisions (20 substeps)
    gravity: float = 9.81                 # [m/s^2]
    nominal_altitude_m: float = 10.0      # Reference cruise altitude [m]
    kp_pos: float = 2.0                   # Horizontal velocity / outer loop proportional gain
    kd_pos: float = 0.8                   # Horizontal velocity / outer loop derivative gain
    ki_pos: float = 0.35                  # Anti-windup integral gain for steady-state tracking
    kp_att: float = 1.0                   # Attitude proportional gain (scaled to torque)
    kd_att: float = 0.1                   # Attitude rate damping gain
    tau_motor: float = 0.030              # 30 ms first-order motor/thrust lag [s]
    max_tilt_rad: float = 0.45            # Attitude envelope (~25.8 deg)
    thrust_margin_ratio: float = 1.85     # Max total thrust / hover weight
    # E8 one-factor disturbance sweeps (Table S22, S26)
    control_delay_ms: float = 0.0         # {0, 40, 80, 120, 160} ms
    comm_noise_std: float = 0.0           # {0.00, 0.05, 0.10, 0.15, 0.20}
    attitude_atten_kappa: float = 0.0     # {0.0, 0.5, 1.0, 1.5, 2.0} -> G_ctrl = 1/(1+kappa)
    eps_utp: float = 1e-6                 # UTP denominator stabilizer
    compensator: CompensatorConfig = field(default_factory=CompensatorConfig)
