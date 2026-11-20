# MTC-MAPPO: Capability-Aware MAPPO with Task-Conditioned Communication Prioritization

Complete reference implementation of the models, environments, training loops, execution-layer controllers, and evaluation/diagnostic suites for the *Aerospace Science and Technology* paper:

> **Capability-aware multi-agent proximal policy optimization with task-conditioned communication prioritization for heterogeneous unmanned aerial vehicle teams**

---

## 1. Repository Architecture

```text
MTC-MAPPO-Code/
├── configs/
│   └── defaults.py              # Tables 1-3, S2, S13, S17, S22, S26, S29, S31, S32, S48, S53-S56 constants
├── envs/
│   ├── wireless.py              # Eqs. (7)-(12), LOS/NLOS/Rician fading, UAV & ground interference, E3 FIFO delay/loss
│   ├── service_and_reward.py    # Eqs. (13)-(21), (23), (41)-(42), ADD/GEO/HARM, E7, S13 alignment, S19 counterfactual
│   ├── lf_uav_env.py            # 2D heterogeneous UAV Dec-POMDP (LF, E1-E9, E-HET, S4, S9, S13, N=5/10/20, state cloning)
│   └── pybullet_uav_env.py      # 6-DOF PyBullet quadrotor (headless DIRECT & GUI), 100 Hz cascaded PID, Group A/B/C compensator
├── models/
│   ├── encoders.py              # Shared capability encoder f_cap + F_C (Eq. 22) & demand encoder F_D (Eq. 24)
│   ├── attention.py             # MTC multi-head Query-Key cosine attention + physical modulation (Eqs. 24-29), Morph-GAT, GAT, TarMAC
│   └── policy.py                # Extended-action PPO density & ratio recomputation (Eqs. 30-37), all baselines/ablations, B2/B3
├── train/
│   ├── buffer.py                # On-policy rollout buffer with raw obs, masks, extended actions, and exact B_t / C_t GAE (Eqs. 39-40)
│   ├── mappo_trainer.py         # CTDE-MAPPO trainer (Algorithm S2, Table S31 clipped value loss & advantage pop-SD norm)
│   ├── train_main.py            # Unified training CLI for all methods, service rules, wireless modes, and ablations
│   └── sweep_ofat.py            # One-factor-at-a-time (OFAT) matching-score sensitivity sweep (Table S48)
├── eval/
│   ├── stats.py                 # Exact 2^n two-sided sign-flip test (Eq. S8), Holm step-down adjustment, paired t-CI, Cohen d_z
│   ├── eval_benchmarks.py       # Closed-loop LF / E1-E5 / E-HET / B2-B3 / S6 active-sender allocation diagnostics
│   ├── eval_masking.py          # LF masking (Fig. 5) & state-cloned Top vs. Distance-matched masking (<=5 m, UAV-ID tracked, S45-S46)
│   ├── eval_interference.py     # Direct prior mixing (S40-S43), 12-frame fixed-allocation radio replay (S42), open-loop S30/S33/S36
│   ├── eval_spearman.py         # Candidate Spearman across D0-D6 (S29), C^ref SD (S49), 7-point support (S50), 50-step return drop (S44)
│   ├── eval_scaling_timing.py   # N=5/10/20 link-count proxies & single-thread CPU timing (1,000 warmup + 10,000 passes, S7, S51-S52)
│   ├── eval_pybullet.py         # E8 PyBullet transfer & count-based retention sweeps (S22-S24), S53 single-UAV recovery, S54-S56 Group A/B/C
│   └── eval_main.py             # Unified CLI dispatcher for all evaluation and diagnostic protocols
├── scripts/
│   ├── run_unit_tests.sh        # Runs full pytest unit & boundary test suite
│   ├── run_smoke_pipeline.sh    # Runs end-to-end training & evaluation smoke pipeline across all modules
│   ├── reproduce_benchmarks.sh  # Full-scale multi-seed training launcher (E1, E2, LF, S48)
│   └── reproduce_pybullet_and_diagnostics.sh # Full-scale frozen-checkpoint & PyBullet evaluation launcher
└── tests/                       # Comprehensive pytest verification suite
```

---

## 2. Environment Setup

Requires Python $\ge 3.8$ with `numpy`, `scipy`, `torch`, `pybullet`, and `pytest`:

```bash
pip install -r requirements.txt
```

All PyBullet simulations default to headless mode (`pybullet.DIRECT`), requiring no X11/GUI display server on Linux workstations or clusters.

---

## 3. Quick Verification (Unit Tests & End-to-End Smoke Pipeline)

1. **Run the complete `pytest` unit test suite** (verifies Eqs. 1–42, empty/single-neighbor boundary cases, power budget projection, Table S7 module parameter counts, PPO extended-action log-density invariance & gradient backpropagation into the attention prior, Table S28 exact $2^{10}$ sign-flip & Holm adjustment, headless 6-DOF PyBullet dynamics, Group A/B/C OU compensator physics, and end-to-end training/evaluation):

```bash
bash scripts/run_unit_tests.sh
```

2. **Run the CLI end-to-end training & evaluation smoke pipeline**:

```bash
bash scripts/run_smoke_pipeline.sh
```

---

## 4. Training & Full Reproduction Usage

### Train a Single Run (`MTC-MAPPO`, Baselines, or Ablations)

```bash
# Primary E1 benchmark (stochastic LOS/NLOS/Rician channel, 1.3e6 joint steps)
python3 -m train.train_main \
  --method mtc_mappo \
  --wireless-mode e1 \
  --service-rule ADD \
  --seed 1001 \
  --total-steps 1300000 \
  --save-path checkpoints/e1/mtc_mappo_seed1001.pt

# Baselines: morph_gat | gat_mappo | mappo_morph | role_mappo | mappo | tarmac_b1
# Ablations: --ablate-cap | --ablate-attn (E2) | --ablate-mod | --ablate-prior
```

### Run Specific Evaluation Suites

```bash
# 1. Distance-matched attention masking (Tables S45-S46)
python3 -m eval.eval_main --suite masking_dist_matched \
  --mtc-ckpts checkpoints/e1/mtc_mappo_seed1001.pt \
  --episodes 200 --output-json outputs/s45_masking.json

# 2. Direct prior-mixing intervention & 12-frame radio-window replay (Tables S40-S43)
python3 -m eval.eval_main --suite prior_mixing \
  --mtc-ckpts checkpoints/e1/mtc_mappo_seed1001.pt \
  --gat-ckpts checkpoints/e1/gat_mappo_seed1001.pt \
  --episodes 200 --output-json outputs/s40_prior_mixing.json

python3 -m eval.eval_main --suite radio_window_s42 \
  --mtc-ckpts checkpoints/e1/mtc_mappo_seed1001.pt \
  --gat-ckpts checkpoints/e1/gat_mappo_seed1001.pt \
  --episodes 200 --output-json outputs/s42_radio_window.json

# 3. Candidate-level Spearman across D0-D6 & 50-step masked return loss (Tables S44, S49-S50)
python3 -m eval.eval_main --suite spearman_s44 \
  --mtc-ckpts checkpoints/e1/mtc_mappo_seed1001.pt \
  --episodes 600 --output-json outputs/s44_spearman.json

# 4. Team-size scaling N=5/10/20 & single-thread CPU latency (Tables S51-S52)
python3 -m eval.eval_main --suite scaling_s51 --wireless-mode lf \
  --mtc-ckpts checkpoints/lf/mtc_mappo_seed0.pt \
  --gat-ckpts checkpoints/lf/gat_mappo_seed0.pt \
  --team-sizes 5 10 20 --episodes 100 \
  --warmup-passes 1000 --timed-passes 10000 \
  --output-json outputs/s51_scaling.json

# 5. PyBullet 6-DOF single-platform recovery (S53) & 5-UAV Group A/B/C compensation (S54-S56)
python3 -m eval.eval_main --suite pybullet_s53_single \
  --episodes 20 --output-json outputs/s53_single.json

python3 -m eval.eval_main --suite pybullet_s54_comp \
  --mtc-ckpts checkpoints/e1/mtc_mappo_seed1001.pt \
  --episodes 100 --output-json outputs/s54_comp.json
```

---

## 5. Module Parameter Accounting Alignment (Supplementary Table S7 & Table S3)

All four mutually exclusive module groups (`Actor`, `Critic`, `Attention`, and `Shared encoders`) match Supplementary Table S7 (and `MAPPO` / `Role-MAPPO` totals match Table S3):

| Method | Actor | Critic | Attention | Shared Encoders | Total |
|---|---:|---:|---:|---:|---:|
| `MAPPO` | — | — | `0` | — | `219,400` |
| `Role-MAPPO` | — | — | `0` | — | `236,900` |
| `MAPPO-morph` | `84,600` | `117,900` | `0` | `18,300` | `220,800` |
| `GAT-MAPPO` | `93,200` | `126,800` | `47,900` | `18,300` | `286,200` |
| `Morph-GAT` | `96,700` | `132,400` | `57,300` | `18,300` | `304,700` |
| **`MTC-MAPPO`** | **`97,600`** | **`132,700`** | **`61,000`** | **`18,300`** | **`309,600`** |
