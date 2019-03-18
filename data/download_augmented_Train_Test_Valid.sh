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

wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part01.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part02.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part03.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part04.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part05.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part06.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part07.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part08.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part09.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part10.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part11.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part12.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part13.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part14.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part15.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part16.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part17.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part18.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part19.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part20.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part21.rar
wget --no-check-certificate https://github.com/monaen/Meng2018Largescale/raw/data/Augmented/Augmented.part22.rar

echo "Unzipping......"

unrar x Augmented.part01.rar && rm -f Augmented.part*.rar

echo "Done."
