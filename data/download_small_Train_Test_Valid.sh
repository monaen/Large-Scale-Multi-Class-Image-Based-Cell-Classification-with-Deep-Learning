#!/usr/bin/env sh
# =================================================================================== #
# | Description:                                                                    | #
# |     Script to download the "Human Somatic Label-free Bright-field Cell          | #
# |     Images" Dataset (small)                                                     | #
# |                                                                                 | #
# | Citation:                                                                       | #
# |     Nan Meng, Edmund Lam, Tsia, Kevin Kin Man, So, Hayden Kwok-Hay, "Human      | #
# |     somatic label-free bright-field cell images", IEEE Dataport, 2018.          | #
# |     [Online]. Available: http://dx.doi.org/10.21227/H2QW97/. Accessed: Mar. 13, | #
# |     2019.                                                                       | #
# |                                                                                 | #
# | Paper:                                                                          | #
# |     Large-scale Multi-class Image-based Cell Classification with Deep Learning  | #
# |     Nan Meng, Edmund Yin Mun Lam, Kevin Kin Man Tsia, and Hayden Kwok Hay So    | #
# |     IEEE Journal of Biomedical and Health Informatics, 2018                     | #
# |                                                                                 | #
# =================================================================================== #

# This scripts downloads the human somatic cell dataset (small) and unzips it.
# DIR = "$( cd "$(dirname "$0")" ; pwd -P )"

echo "Downloading......"

wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Dataset1/small.part1.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Dataset1/small.part2.rar

echo "Unzipping......"

unrar x small.part1.rar && rm -f small.part*.rar

echo "Done."