#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-outputs/smoke_test}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}/ckpts" "${OUT_DIR}/eval"

echo "[1/8] Training smoke checkpoints (MTC-MAPPO & GAT-MAPPO)..."
"${PYTHON_BIN}" -m train.train_main \
  --method mtc_mappo --wireless-mode e1 --seed 1001 \
  --total-steps 64 --rollout-steps 32 --minibatch-size 32 --max-ep-steps 20 \
  --save-path "${OUT_DIR}/ckpts/mtc_1001.pt"

"${PYTHON_BIN}" -m train.train_main \
  --method gat_mappo --wireless-mode e1 --seed 1001 \
  --total-steps 64 --rollout-steps 32 --minibatch-size 32 --max-ep-steps 20 \
  --save-path "${OUT_DIR}/ckpts/gat_1001.pt"

echo "[2/8] Running OFAT matching-score sensitivity smoke (Table S48)..."
"${PYTHON_BIN}" -m train.sweep_ofat \
  --seeds 0 --total-steps 32 --rollout-steps 16 --eval-episodes 2 --max-ep-steps 16 \
  --labels default tau_a=0.25 \
  --output-dir "${OUT_DIR}/eval/s48_ofat"

echo "[3/8] Evaluating benchmark & B2/B3 comparators (Tables 4, 8, 9)..."
"${PYTHON_BIN}" -m eval.eval_main \
  --suite benchmark --mtc-ckpts "${OUT_DIR}/ckpts/mtc_1001.pt" \
  --episodes 3 --max-ep-steps 20 --output-json "${OUT_DIR}/eval/bench_mtc.json"

"${PYTHON_BIN}" -m eval.eval_main \
  --suite benchmark --comparator b2 --mtc-ckpts "${OUT_DIR}/ckpts/mtc_1001.pt" \
  --episodes 3 --max-ep-steps 20 --output-json "${OUT_DIR}/eval/bench_b2.json"

"${PYTHON_BIN}" -m eval.eval_main \
  --suite benchmark --comparator b3 --mtc-ckpts "${OUT_DIR}/ckpts/mtc_1001.pt" \
  --episodes 3 --max-ep-steps 20 --output-json "${OUT_DIR}/eval/bench_b3.json"

echo "[4/8] Evaluating distance-matched attention masking (Tables S45-S46)..."
"${PYTHON_BIN}" -m eval.eval_main \
  --suite masking_dist_matched --mtc-ckpts "${OUT_DIR}/ckpts/mtc_1001.pt" \
  --episodes 4 --max-ep-steps 20 --output-json "${OUT_DIR}/eval/masking_s45.json"

echo "[5/8] Evaluating prior mixing, 12-frame radio replay & open-loop wireless (S30/S33/S36/S40-S43)..."
"${PYTHON_BIN}" -m eval.eval_main \
  --suite prior_mixing \
  --mtc-ckpts "${OUT_DIR}/ckpts/mtc_1001.pt" --gat-ckpts "${OUT_DIR}/ckpts/gat_1001.pt" \
  --episodes 2 --max-ep-steps 20 --output-json "${OUT_DIR}/eval/prior_mixing_s40.json"

"${PYTHON_BIN}" -m eval.eval_main \
  --suite radio_window_s42 \
  --mtc-ckpts "${OUT_DIR}/ckpts/mtc_1001.pt" --gat-ckpts "${OUT_DIR}/ckpts/gat_1001.pt" \
  --episodes 2 --max-ep-steps 20 --output-json "${OUT_DIR}/eval/radio_window_s42.json"

"${PYTHON_BIN}" -m eval.eval_main \
  --suite open_loop_wireless \
  --output-json "${OUT_DIR}/eval/open_loop_s19_s30_s33_s36.json"

echo "[6/8] Evaluating candidate Spearman & 50-step masked return loss (Tables S44, S49, S50)..."
"${PYTHON_BIN}" -m eval.eval_main \
  --suite spearman_s44 --mtc-ckpts "${OUT_DIR}/ckpts/mtc_1001.pt" \
  --episodes 12 --max-ep-steps 20 --output-json "${OUT_DIR}/eval/spearman_s44.json"

echo "[7/8] Evaluating N=5/10/20 team-size scaling & single-thread CPU timing (Tables S51-S52)..."
"${PYTHON_BIN}" -m eval.eval_main \
  --suite scaling_s51 \
  --mtc-ckpts "${OUT_DIR}/ckpts/mtc_1001.pt" --gat-ckpts "${OUT_DIR}/ckpts/gat_1001.pt" \
  --team-sizes 5 10 20 --episodes 2 --max-ep-steps 16 \
  --warmup-passes 5 --timed-passes 20 \
  --output-json "${OUT_DIR}/eval/scaling_s51.json"

echo "[8/8] Evaluating headless PyBullet 6-DOF transfer & Group A/B/C compensator (Tables 11-12, S53-S56)..."
"${PYTHON_BIN}" -m eval.eval_main \
  --suite pybullet_s54_comp --mtc-ckpts "${OUT_DIR}/ckpts/mtc_1001.pt" \
  --episodes 2 --max-ep-steps 10 --output-json "${OUT_DIR}/eval/pybullet_s54.json"

echo "=== All End-to-End Smoke Pipeline Stages Completed Successfully! ==="
