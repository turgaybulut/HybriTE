#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
RESET='\033[0m'

if [ $# -eq 0 ]; then
    echo -e "${CYAN}Usage:${RESET} $0 <base_directory>"
    echo -e "${CYAN}Example:${RESET} $0 models/human/hybrite"
    exit 1
fi

BASE_DIR="$1"

if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}${BOLD}Error:${RESET} ${RED}Directory '$BASE_DIR' does not exist${RESET}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREDICT_SCRIPT="$SCRIPT_DIR/predict.py"

if [ ! -f "$PREDICT_SCRIPT" ]; then
    echo -e "${RED}${BOLD}Error:${RESET} ${RED}predict.py not found at $PREDICT_SCRIPT${RESET}"
    exit 1
fi

echo -e "${BOLD}${MAGENTA}Starting prediction for all folds in: ${CYAN}$BASE_DIR${RESET}\n"

for FOLD_DIR in "$BASE_DIR"/fold_*; do
    if [ ! -d "$FOLD_DIR" ]; then
        continue
    fi

    FOLD_NAME=$(basename "$FOLD_DIR")
    echo -e "${BOLD}${BLUE}==============================================================================${RESET}"
    echo -e "${BOLD}${CYAN}Processing ${MAGENTA}$FOLD_NAME${RESET}${CYAN}...${RESET}"
    echo -e "${BOLD}${BLUE}==============================================================================${RESET}"

    HPARAMS_FILE="$FOLD_DIR/hparams.yaml"
    if [ ! -f "$HPARAMS_FILE" ]; then
        echo -e "${YELLOW}  Warning:${RESET} ${YELLOW}hparams.yaml not found in $FOLD_DIR, skipping...${RESET}"
        continue
    fi

    CHECKPOINT_DIR="$FOLD_DIR/checkpoints"
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo -e "${YELLOW}  Warning:${RESET} ${YELLOW}checkpoints directory not found in $FOLD_DIR, skipping...${RESET}"
        continue
    fi

    CHECKPOINT_FILE=$(find "$CHECKPOINT_DIR" -name "*.ckpt" | head -n 1)
    if [ -z "$CHECKPOINT_FILE" ]; then
        echo -e "${YELLOW}  Warning:${RESET} ${YELLOW}No .ckpt file found in $CHECKPOINT_DIR, skipping...${RESET}"
        continue
    fi

    echo -e "${CYAN}  Checkpoint:${RESET} $CHECKPOINT_FILE"
    echo -e "${CYAN}  Config:${RESET} $HPARAMS_FILE"
    echo -e "${CYAN}  Output:${RESET} $FOLD_DIR"
    echo ""

    echo -e "${CYAN}  Running prediction...${RESET}"
    python "$PREDICT_SCRIPT" \
        --checkpoint "$CHECKPOINT_FILE" \
        --config "$HPARAMS_FILE" \
        --output_dir "$FOLD_DIR" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  Successfully processed $FOLD_NAME${RESET}"
    else
        echo -e "${RED}  Failed to process $FOLD_NAME${RESET}"
    fi
    echo ""
done

echo -e "${BOLD}${GREEN}==============================================================================${RESET}"
echo -e "${BOLD}${GREEN}Done processing all folds in ${CYAN}$BASE_DIR${RESET}"
echo -e "${BOLD}${GREEN}==============================================================================${RESET}"
echo ""

echo -e "${BOLD}${MAGENTA}Creating cross-validation summary...${RESET}\n"

set -euo pipefail

CV_SUMMARY_FILE="${BASE_DIR}/cross_validation_summary.csv"
TEMP_DIR=$(mktemp -d)

trap 'rm -rf "$TEMP_DIR"' EXIT

FOLD_DIRS=($(find "$BASE_DIR" -maxdepth 1 -type d -name "fold_*" | sort))

if [ ${#FOLD_DIRS[@]} -eq 0 ]; then
    echo -e "${RED}${BOLD}Error:${RESET} ${RED}No fold directories found in $BASE_DIR${RESET}" >&2
    exit 1
fi

for fold_dir in "${FOLD_DIRS[@]}"; do
    metrics_file="${fold_dir}/metrics.csv"
    if [ ! -f "$metrics_file" ]; then
        echo -e "${YELLOW}  Warning:${RESET} ${YELLOW}metrics.csv not found in $fold_dir${RESET}" >&2
        continue
    fi

    fold_name=$(basename "$fold_dir")
    tail -n +2 "$metrics_file" | grep -v '^[[:space:]]*$' | awk -F',' -v fold="$fold_name" '{print $1","fold","$2}' >> "${TEMP_DIR}/all_metrics.csv"
done

if [ ! -f "${TEMP_DIR}/all_metrics.csv" ]; then
    echo -e "${RED}${BOLD}Error:${RESET} ${RED}No metrics data collected${RESET}" >&2
    exit 1
fi

awk -F',' -v temp_dir="$TEMP_DIR" '
BEGIN {
    n_folds = 0
}
{
    metric = $1
    fold = $2
    value = $3

    metrics[metric] = 1
    values[metric, fold] = value

    if (!(fold in fold_order)) {
        fold_order[fold] = n_folds
        fold_names[n_folds] = fold
        n_folds++
    }
}
END {
    for (metric in metrics) {
        sum = 0
        sum_sq = 0
        count = 0
        min_val = ""
        max_val = ""

        for (i = 0; i < n_folds; i++) {
            fold = fold_names[i]
            if ((metric, fold) in values) {
                val = values[metric, fold]
                sum += val
                sum_sq += val * val
                count++

                if (min_val == "" || val < min_val) min_val = val
                if (max_val == "" || val > max_val) max_val = val
            }
        }

        if (count > 0) {
            mean = sum / count
            variance = (sum_sq / count) - (mean * mean)
            std = sqrt(variance > 0 ? variance : 0)

            printf "%s,%.10f,%.10f,%.10f,%.10f", metric, mean, std, min_val, max_val

            for (i = 0; i < n_folds; i++) {
                fold = fold_names[i]
                if ((metric, fold) in values) {
                    printf ",%.10f", values[metric, fold]
                } else {
                    printf ","
                }
            }
            printf "\n"
        }
    }
}' "${TEMP_DIR}/all_metrics.csv" > "${TEMP_DIR}/summary_body.csv"

printf "metric,mean,std,min,max" > "$CV_SUMMARY_FILE"
for fold_dir in "${FOLD_DIRS[@]}"; do
    printf ",$(basename "$fold_dir")" >> "$CV_SUMMARY_FILE"
done
printf "\n" >> "$CV_SUMMARY_FILE"

sort "${TEMP_DIR}/summary_body.csv" >> "$CV_SUMMARY_FILE"

echo -e "${GREEN}  Cross-validation summary created: ${CYAN}$CV_SUMMARY_FILE${RESET}"
