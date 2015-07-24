#!/usr/bin/env bash
# This script downloads SVHN dataset and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading training data..."

wget http://ufldl.stanford.edu/housenumbers/train.tar.gz

echo "Downloading testing data..."

wget http://ufldl.stanford.edu/housenumbers/test.tar.gz

echo "Unzipping..."

tar -xf train.tar.gz && rm -f train.tar.gz
tar -xf test.tar.gz && rm -f test.tar.gz

echo "Downloading json data..."
wget http://nicodjimenez.com/data/svhn/test.json
wget http://nicodjimenez.com/data/svhn/train.json

echo "Done."
