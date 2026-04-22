#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env (works with bash)
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

dataset_name="onion"
drop_num=5
drop_rating=-1
fairness_user_method="activity"

data_path="${DATASET_PATH:-$PROJECT_ROOT/dataset}"
data_csv="$data_path/onion/data.csv"

if [[ ! -f "$data_csv" ]]; then
  echo "Error: $data_csv not found."
  echo "Run data/preprocess_onion.ipynb first to create data.csv from the Music4All raw data."
  exit 1
fi

python data/process_data.py \
    --dataset_name "$dataset_name" \
    --data_path "$data_path" \
    --drop_num "$drop_num" \
    --drop_rating "$drop_rating" \
    --enable_fairness \
    --create_recbole \
    --fairness_user_method "$fairness_user_method"\
    --file_name "data.csv"
