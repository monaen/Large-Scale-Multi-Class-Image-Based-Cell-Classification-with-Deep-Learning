#!/usr/bin/env sh
# ============================================================================== #
# |
# |
# |
# |
# |
# ============================================================================== #

# This scripts downloads the human somatic cell dataset (small) and unzips it.
DIR = "$( cd "$(dirname "$0")" ; pwd -P )"

echo "Downloading......"

wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/small.part1.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/small.part2.rar

echo "Unzipping"

# tar -xf
