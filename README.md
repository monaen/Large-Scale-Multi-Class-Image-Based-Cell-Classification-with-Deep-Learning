## Large-scale Multi-class Image-based Cell Classification with Deep Learning (Tensorflow Implementation)

The tensorflow project for work "Large-scale Multi-class Image-based Cell Classification with Deep Learning"

### Environment
* Ubuntu 16.04
* Python 2.7

### Requirements
* notebook==5.7.4
* numpy==1.15.4
* Pillow==5.3.0
* scikit-image==0.14.1
* scikit-learn==0.20.1
* scipy==1.1.0
* opencv>=2.4.12
* tensorboard>=1.4.0
* tensorflow>=1.4.0
* tqdm==4.28.1

### Dataset
We provide three different datasets, and all the datasets could be downloaded at [https://github.com/monaen/Meng2018Largescale/tree/data](https://github.com/monaen/Meng2018Largescale/tree/data)

* download the small data
```commandline
git clone https://github.com/monaen/Meng2018Largescale.git --branch code --single-branch
cd Meng2018Largescale/data
./download_small_Train_Test_Valid.sh
```

* download the large data
```commandline
./download_large_Train_Test_Valid.sh
```
* download the augmented data
```commandline
./download_augmented_Train_Test_Valid.sh
```

### Reference

Data
```
@data{h2qw97-18,
  doi = {10.21227/H2QW97},
  url = {http://dx.doi.org/10.21227/H2QW97},
  author = {Nan Meng; Edmund Lam; Tsia; Kevin Kin Man; So; Hayden Kwok-Hay },
  publisher = {IEEE Dataport},
  title = {Human somatic label-free bright-field cell images},
  year = {2018}
}
```

Paper
```
@article{Meng2018Largescale,
  title = {Large-scale Multi-class Image-based Cell Classification with Deep Learning},
  author = {Meng, Nan and Lam, Edmund and Tsia, Kevin Kin Man and So, Hayden Kwok-Hay},
  journal = {IEEE journal of biomedical and health informatics},
  year = {2018},
  publisher = {IEEE}
}
