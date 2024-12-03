#!/bin/bash

# List of models to run the Python script with
models=(
    "builtin.psiformer"
    "LinearHF"
    "mlp_relu"
    "mlp_sin"
    "interaction_mlp_sigmoid"
)

# Loop through each model and execute the Python command
for model in "${models[@]}"; do
    echo "Running python script with model: $model"
    python run.py "$model"
done

echo "All models processed."
