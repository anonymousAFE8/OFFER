#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HSTU_ROOT="$REPO_ROOT/hstu_3stage"

RECALL_PORT="${RECALL_PORT:-12355}"
COARSE_PORT="${COARSE_PORT:-12356}"
FINE_PORT="${FINE_PORT:-12357}"
TOP_M="${TOP_M:-50}"

cd "$HSTU_ROOT"
mkdir -p artifacts/checkpoints artifacts/exports artifacts/logs

python main.py --gin_config_file=configs/ml-1m/release-recall-050.gin --master_port="$RECALL_PORT"

python scripts/ml-1m/build_coarse_rank_dataset.py \
  --base_csv ../data/ml-1m/regen-050/sasrec_format_final_recall.csv \
  --recall_topk_path artifacts/exports/release_recall_top1000_050.pt \
  --output_csv artifacts/exports/ml1m_coarse_top1000_050.csv \
  --candidate_k 1000 \
  --drop_users_without_target

python main.py --gin_config_file=configs/ml-1m/release-coarse-050.gin --master_port="$COARSE_PORT"

python scripts/ml-1m/build_fine_rank_dataset.py \
  --base_csv ../data/ml-1m/regen-050/sasrec_format_final_recall.csv \
  --rank_export_path artifacts/exports/release_coarse_top1000_050.pt \
  --output_csv artifacts/exports/ml1m_fine_top50_050.csv \
  --top_m "$TOP_M" \
  --drop_users_without_positive

python main.py --gin_config_file=configs/ml-1m/release-fine-050.gin --master_port="$FINE_PORT"
