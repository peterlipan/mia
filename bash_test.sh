#!/bin/bash

# Define the models and seeds
seeds=(1 42)
atlases=("cc400" "cc200" "aal")

# Iterate over each model and seed
for seed in "${seeds[@]}"; do
    for atlas in "${atlases[@]}"; do
        python3 main.py --debug --seed "$seed" --atlas "$atlas"
    done
done
