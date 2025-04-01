#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=4
device=""

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/friends \
        --words 100 \
        --checkpoint $models/LSTM_dp_0.9.pt \
        --mps \
        --outf $samples/sample_friends_LSTM_dp_0.9
)
