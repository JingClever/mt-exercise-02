#!/bin/bash

scripts=$(dirname "$0")
base=$(realpath "$scripts/..")

logs="$base/logs"
results="$base/results"


mkdir -p "$results"


python "$scripts/plot_ppl.py" \
    --model LSTM \
    --input_dir "$logs" \
    --output "$results/LSTM_ppl.tsv" \
    --plot_output "$results/LSTM_ppl.png" \
    --bar_output "$results/LSTM_test_ppl.png"


python "$scripts/plot_ppl_test.py" \
    --model LSTM \
    --input_dir "$logs" \
    --output_plot "$results/LSTM_test_ppl.png" 

echo "The final result for each model and plot are stored in the results directory."