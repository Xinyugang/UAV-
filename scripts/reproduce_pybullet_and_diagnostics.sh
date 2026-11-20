#!/usr/bin/env bash
set -euo pipefail

# Full-scale frozen-checkpoint evaluation launcher for Masking, Wireless Replay, Spearman, Scaling & PyBullet
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints}"
OUT_ROOT="${OUT_ROOT:-outputs/full_eval}"
DEVICE="${DEVICE:-cpu}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MTC_E1_5=(
  "${CKPT_ROOT}/e1/mtc_mappo_seed1001.pt"
  "${CKPT_ROOT}/e1/mtc_mappo_seed1002.pt"
  "${CKPT_ROOT}/e1/mtc_mappo_seed1003.pt"
  "${CKPT_ROOT}/e1/mtc_mappo_seed1004.pt"
  "${CKPT_ROOT}/e1/mtc_mappo_seed1005.pt"
)
GAT_E1_5=(
  "${CKPT_ROOT}/e1/gat_mappo_seed1001.pt"
  "${CKPT_ROOT}/e1/gat_mappo_seed1002.pt"
  "${CKPT_ROOT}/e1/gat_mappo_seed1003.pt"
  "${CKPT_ROOT}/e1/gat_mappo_seed1004.pt"
  "${CKPT_ROOT}/e1/gat_mappo_seed1005.pt"
)
MTC_E1_10=( "${MTC_E1_5[@]}"
  "${CKPT_ROOT}/e1/mtc_mappo_seed1006.pt"
  "${CKPT_ROOT}/e1/mtc_mappo_seed1007.pt"
  "${CKPT_ROOT}/e1/mtc_mappo_seed1008.pt"
  "${CKPT_ROOT}/e1/mtc_mappo_seed1009.pt"
  "${CKPT_ROOT}/e1/mtc_mappo_seed1010.pt"
)
MTC_LF_5=(
  "${CKPT_ROOT}/lf/mtc_mappo_seed0.pt"
  "${CKPT_ROOT}/lf/mtc_mappo_seed1.pt"
  "${CKPT_ROOT}/lf/mtc_mappo_seed2.pt"
  "${CKPT_ROOT}/lf/mtc_mappo_seed3.pt"
  "${CKPT_ROOT}/lf/mtc_mappo_seed4.pt"
)
GAT_LF_5=(
  "${CKPT_ROOT}/lf/gat_mappo_seed0.pt"
  "${CKPT_ROOT}/lf/gat_mappo_seed1.pt"
  "${CKPT_ROOT}/lf/gat_mappo_seed2.pt"
  "${CKPT_ROOT}/lf/gat_mappo_seed3.pt"
  "${CKPT_ROOT}/lf/gat_mappo_seed4.pt"
)

"${PYTHON_BIN}" -m eval.eval_main --suite open_loop_wireless --output-json "${OUT_ROOT}/open_loop_s19_s30_s33_s36.json"
"${PYTHON_BIN}" -m eval.eval_main --suite prior_mixing --mtc-ckpts "${MTC_E1_5[@]}" --gat-ckpts "${GAT_E1_5[@]}" --episodes 200 --device "${DEVICE}" --output-json "${OUT_ROOT}/s40_s41_s43_prior_mixing.json"
"${PYTHON_BIN}" -m eval.eval_main --suite radio_window_s42 --mtc-ckpts "${MTC_E1_5[@]}" --gat-ckpts "${GAT_E1_5[@]}" --episodes 200 --device "${DEVICE}" --output-json "${OUT_ROOT}/s42_radio_window.json"
"${PYTHON_BIN}" -m eval.eval_main --suite spearman_s44 --mtc-ckpts "${MTC_E1_10[@]}" --episodes 600 --device "${DEVICE}" --output-json "${OUT_ROOT}/s44_s49_s50_spearman.json"
"${PYTHON_BIN}" -m eval.eval_main --suite masking_dist_matched --mtc-ckpts "${MTC_E1_10[@]}" --episodes 200 --device "${DEVICE}" --output-json "${OUT_ROOT}/s45_s46_masking.json"
"${PYTHON_BIN}" -m eval.eval_main --suite scaling_s51 --wireless-mode lf --mtc-ckpts "${MTC_LF_5[@]}" --gat-ckpts "${GAT_LF_5[@]}" --episodes 100 --warmup-passes 1000 --timed-passes 10000 --device "${DEVICE}" --output-json "${OUT_ROOT}/s51_s52_scaling.json"
"${PYTHON_BIN}" -m eval.eval_main --suite pybullet_s53_single --episodes 20 --output-json "${OUT_ROOT}/s53_single_platform.json"
"${PYTHON_BIN}" -m eval.eval_main --suite pybullet_s54_comp --mtc-ckpts "${MTC_E1_5[@]}" --episodes 100 --device "${DEVICE}" --output-json "${OUT_ROOT}/s54_s56_pybullet_comp.json"
