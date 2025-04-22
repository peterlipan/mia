#!/bin/bash

# Define the models and seeds
omegas=(0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0)

# Iterate over each model and seed
for omega in "${omegas[@]}"; do
    python3 main_ttpl.py --debug --ttpl_ratio "$omega"
done

