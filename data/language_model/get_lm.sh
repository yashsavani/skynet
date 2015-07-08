#!/usr/bin/env sh
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget russellsstewart.com/s/lm/vocab.pkl
wget russellsstewart.com/s/lm/train_indices.txt
wget russellsstewart.com/s/lm/valid_indices.txt
wget russellsstewart.com/s/lm/test_indices.txt

echo "Done."
