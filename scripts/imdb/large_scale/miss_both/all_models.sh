#!/bin/bash

# Generate FL data settings - 100 clients
bash script/imdb/large_scale/miss_both/100cl.sh

# All algorithms
# algorithms=("Fed-Prime" "Fed-Intra" "Fed-Inter" "FedAvg_P" "FedProx_P" "FedMSplit_P" "Centralized_P")
algorithms=("Fed-Prime" "FedMSplit_P")

# Define proportions to test
proportions=(0.1 0.2 0.3 0.4 0.5)

for proportion in "${proportions[@]}"; do
    echo "Testing with proportion: $proportion"
    
    for algo in "${algorithms[@]}"; do
        echo "Running algorithm: $algo"
        python main.py \
            --task imdb_cnum100_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 \
            --model "$algo" \
            --algorithm "multimodal.imdb.$algo" \
            --sample uniform \
            --aggregate other \
            --num_rounds 250 \
            --proportion "$proportion" \
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
    
    echo "Finished testing proportion: $proportion"
    echo "========================================"
done