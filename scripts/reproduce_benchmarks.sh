#!/usr/bin/env bash
set -euo pipefail

# Full-scale multi-seed training launcher reproducing Manuscript Tables 4-8 & Supplementary S8, S11, S20, S38, S39, S48
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints}"
DEVICE="${DEVICE:-cpu}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=== [1] Primary E1 Benchmark: 6 Methods x 10 Matched Seeds (1001..1010) ==="
for METHOD in mappo role_mappo mappo_morph gat_mappo morph_gat mtc_mappo; do
  for SEED in $(seq 1001 1010); do
    "${PYTHON_BIN}" -m train.train_main \
      --method "${METHOD}" --wireless-mode e1 --seed "${SEED}" \
      --total-steps 1300000 --device "${DEVICE}" \
      --save-path "${CKPT_ROOT}/e1/${METHOD}_seed${SEED}.pt"
  done
done

echo "=== [2] E2 Primary Task-Conditioned-Attention Ablation (Seeds 1001..1010) ==="
for SEED in $(seq 1001 1010); do
  "${PYTHON_BIN}" -m train.train_main \
    --method mtc_mappo --ablate-attn --wireless-mode e1 --seed "${SEED}" \
    --total-steps 1300000 --device "${DEVICE}" \
    --save-path "${CKPT_ROOT}/e2/mtc_wo_attn_seed${SEED}.pt"
done

echo "=== [3] LF Cohort & Table 7 Component Ablations ==="
for METHOD in gat_mappo morph_gat mtc_mappo; do
  for SEED in $(seq 0 9); do
    "${PYTHON_BIN}" -m train.train_main \
      --method "${METHOD}" --wireless-mode lf --seed "${SEED}" \
      --total-steps 1300000 --device "${DEVICE}" \
      --save-path "${CKPT_ROOT}/lf/${METHOD}_seed${SEED}.pt"
  done
done
for METHOD in mappo role_mappo mappo_morph; do
  for SEED in $(seq 0 4); do
    "${PYTHON_BIN}" -m train.train_main \
      --method "${METHOD}" --wireless-mode lf --seed "${SEED}" \
      --total-steps 1300000 --device "${DEVICE}" \
      --save-path "${CKPT_ROOT}/lf/${METHOD}_seed${SEED}.pt"
  done
done

echo "=== [4] Table S48 OFAT Matching-Score Sensitivity Sweep ==="
"${PYTHON_BIN}" -m train.sweep_ofat \
  --seeds 0 1 2 3 4 --total-steps 1300000 \
  --output-dir "outputs/s48_ofat" --device "${DEVICE}"
