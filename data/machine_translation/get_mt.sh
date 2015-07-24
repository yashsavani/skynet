#!/usr/bin/env sh
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading processed data"

wget russellsstewart.com/s/mt/build.tgz
tar -xf build.tgz

echo "Done."
