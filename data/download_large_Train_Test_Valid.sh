#!/usr/bin/env sh
# =================================================================================== #
# | Description:                                                                    | #
# |     Script to download the "Human Somatic Label-free Bright-field Cell          | #
# |     Images" Dataset (large)                                                     | #
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

# This scripts downloads the human somatic cell dataset (large) and unzips it.
# DIR = "$( cd "$(dirname "$0")" ; pwd -P )"

echo "Downloading......"

wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part01.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part02.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part03.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part04.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part05.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part06.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part07.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part08.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part09.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part10.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part11.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part12.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part13.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part14.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part15.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/large.part16.rar

echo "Unzipping......"

unrar x large.part01.rar && rm -f large.part*.rar

echo "Done."