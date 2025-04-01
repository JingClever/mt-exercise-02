#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
logs=$base/logs

mkdir -p $models
mkdir -p $logs


num_threads=4
device=""

SECONDS=0

# Parameters setting for LSTM & GRU
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main_ex2.py --data $data/friends \
        --epochs 50 \
        --model LSTM \
        --log-interval 100 \
        --emsize 256 --nhid 256 --dropout 0.8 --tied \
        --mps \
        --save $models/LSTM_dp_0.8.pt \
        --ppl-log $logs/LSTM_ppl_dp_0.8.tsv
)

# (cd $tools/pytorch-examples/word_language_model &&
#     CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main_ex2.py --data $data/friends \
#         --epochs 50 \
#         --model GRU \
#         --log-interval 100 \
#         --emsize 256 --nhid 256 --dropout 0.2 --tied \
#         --mps \
#         --save $models/GRU_dp_0.2.pt \
#         --ppl-log $logs/GRU_ppl_dp_0.2.tsv
# )
# # Parameters setting for Transformer, RNN_RELU, RNN_TANH
# (cd $tools/pytorch-examples/word_language_model &&
#     CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main_ex2.py --data $data/friends \
#         --epochs 50 \
#         --model RNN_TANH \
#         --lr 1.0 \
#         --log-interval 100 \
#         --emsize 256 --nhid 256 --dropout 0.9 \
#         --nhead 4 \
#         --mps \
#         --save $models/RNN_TANH_dp_0.9.pt \
#         --ppl-log $logs/RNN_TANH_ppl_dp_0.9.tsv
# )



echo "time taken:"
echo "$SECONDS seconds"
