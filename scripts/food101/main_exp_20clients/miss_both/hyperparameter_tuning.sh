#!/bin/bash

# Generate FL data settings - 20 clients first
bash script/food101/main_exp_20clients/miss_both/20cl.sh

# Define the hyperparameter ranges
pool_sizes=(10 15 20 25 30 35 40)
top_k_values=(5 6 7 8 9 10)
outer_loops=(5 6 7 8 9 10)

# All algorithms
algorithms=("Fed-Prime" "Fed-Intra" "Fed-Inter" "FedAvg_P" "FedProx_P" "FedMSplit_P" "Centralized_P")
# algorithms=("Fed-Prime")


for algo in "${algorithms[@]}"; do
    for pool_size in "${pool_sizes[@]}"; do
        for top_k in "${top_k_values[@]}"; do
            for num_outer_loops in "${outer_loops[@]}"; do
                echo "Running algorithm: $algo with pool_size=$pool_size, top_k=$top_k, num_outer_loops=$num_outer_loops"
                
                python main.py \
                    --task food101_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 \
                    --model "$algo" \
                    --algorithm "multimodal.food101.$algo" \
                    --sample full \
                    --aggregate other \
                    --num_rounds 250 \
                    --proportion 1.0 \
                    --lr_scheduler 0 \
                    --seed 1234 \
                    --learning_rate 0.05 \
                    --num_epochs 1 \
                    --pool_size $pool_size \
                    --top_k $top_k \
                    --num_outer_loops $num_outer_loops \
                    --learning_rate_decay 1.0 \
                    --batch_size 64 \
                    --test_batch_size 64 \
                    --max_text_len 40 \
                    --gpu 2 \
                    --wandb

                echo "Finished running $algo with current hyperparameters"
                echo "----------------------------------------"
            done
        done
    done
done 