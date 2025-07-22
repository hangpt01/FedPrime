#!/bin/bash

# Generate FL data settings - 20 clients first
bash script/imdb/main_exp_20clients/miss_text/20cl.sh

# All algorithms
algorithms=("Fed-Prime" "Fed-Intra" "Fed-Inter" "FedAvg_P" "FedMSplit_P" "Centralized_P")

# Proposal-related algorithms
# algorithms=("Fed-Prime" "Fed-Intra" "Fed-Inter")

# Baseline algorithms
# algorithms=("FedAvg_P" "FedMSplit_P")

# Centralized algorithm
# algorithms=("Centralized_P")

for algo in "${algorithms[@]}"; do
    echo "Running algorithm: $algo"
    python main.py \
        --task imdb_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_text_text_both_ratio_0.0 \
        --model "$algo" \
        --algorithm "multimodal.imdb.$algo" \
        --sample full \
        --aggregate other \
        --num_rounds 250 \
        --proportion 1.0 \
        --lr_scheduler 0 \
        --seed 1234 \
        --learning_rate 0.05 \
        --num_epochs 250 \
        --pool_size 20 \
        --top_k 5 \
        --num_outer_loops 5 \
        --learning_rate_decay 1.0 \
        --batch_size 256 \
        --test_batch_size 256 \
        --max_text_len 128 \
        --gpu 0 \
        --wandb

    echo "Finished running $algo"
    echo "----------------------------------------"
done