#!/bin/bash

# List of models to run the Python script with
models=(
  "dummy"
  "gru"
  "lstm"
  "mlp"
  "resnet"
  "vanila_rnn"
)

# Loop through each model and execute the Python command
for model in "${models[@]}"; do
    echo "Running python script with model: $model"
    python run.py general_with_cusp --steps 2000 --features naive_gnn --model $model
done

echo "All models processed."
