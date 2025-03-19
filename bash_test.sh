#!/bin/bash

# Define the models and seeds
seeds=(1 42 123 456 789)
atlases=("cc400" "cc200" "aal")

# Iterate over each model and seed
for seed in "${seeds[@]}"; do
    for atlas in "${atlases[@]}"; do
        python3 main.py --seed "$seed" --atlas "$atlas"
    done
done
