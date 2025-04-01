#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!


echo "Downloading Friends dataset..."
mkdir -p $data/friends
mkdir -p $data/friends/raw

python3 $base/scripts/download_friends.py 
# Cut to first 8k lines due to its big size
head -n 8000 $base/data/friends/raw/friends.txt > $base/data/friends/raw/friends_8k.txt
cat $data/friends/raw/friends_8k.txt | python $base/scripts/preprocess_raw.py > $base/data/friends/raw/friends.cleaned.txt

# tokenize, fix vocabulary upper bound
cat $data/friends/raw/friends.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/friends/raw/friends.preprocessed.txt

# split into train, valid and test


head -n 10497 $data/friends/raw/friends.preprocessed.txt | tail -n 1200 > $data/friends/valid.txt
head -n 11697 $data/friends/raw/friends.preprocessed.txt | tail -n 1200 > $data/friends/test.txt
tail -n 11697 $data/friends/raw/friends.preprocessed.txt | tail -n 9297 > $data/friends/train.txt