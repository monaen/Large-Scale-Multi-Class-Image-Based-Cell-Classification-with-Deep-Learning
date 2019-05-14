# Large-scale Multi-class Image-based Cell Classification with Deep Learning [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
Tensorflow implementation of "Large-scale Multi-class Image-based Cell Classification with Deep Learning" by Nan M., Edmund Y. L., Tsia K. M., Hayden K-H. So. [[Paper]](https://monaen.github.io/Meng2018Largescale/materials/Largescale_Multiclass_Imagebased_Cell_Classification_with_Deep_Learning.pdf)

### Project page
[https://monaen.github.io/Meng2018Largescale](https://monaen.github.io/Meng2018Largescale)

### Prerequisites
This codebase was developed and tested with Tensorflow 1.8.0

### Environment
* Ubuntu 14.04/16.04
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

### Usage
1. clone the code branch of the project (Notice: we do not suggest to clone the entire project)
```commandline
git clone https://github.com/monaen/Meng2018Largescale.git --branch code --single-branch
```

2. head to the `data` folder and download the data
```commandline
cd Meng2018Largescale/data
./download_augmented_Train_Test_Valid.sh
```

3. train (test) the model
```commandline
python main.py --dataset "data/Augmented" --batchsize 2000 --imgsize 128
```

4. run the demo
```commandline
python demo.py
```

### Reference

Data
```
@data{h2qw97-18,
  doi = {10.21227/H2QW97},
  url = {http://dx.doi.org/10.21227/H2QW97},
  author = {Meng, Nan and Lam, Edmund and Tsia, Kevin Kin Man and So, Hayden Kwok-Hay},
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
```

### TODO
- [x] Update the proposed model -- CellNet
- [x] Upload the small/large dataset
- [x] Upload the augmented dataset
- [x] Upload the well-trained weights for CellNet
- [x] Add the Hep2 model from [Gao et al.("HEp-2 cell image classification with deep convolutional neural networks")](https://ieeexplore.ieee.org/document/7400923)
- [x] Upload the well-trained weights for Gao et al.'s model.
- [x] Add the Hep2 model from [Liu et al.("HEp-2 cell classification based on a Deep Autoencoding-Classification convolutional neural network")](https://ieeexplore.ieee.org/document/7950689)
- [x] Upload the well-trained weights for Liu et al.'s model.
- [x] Add the model from [Oei et al.("Convolutional neural network for cell classification using microscope images of intracellular actin networks.")](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0213626)
- [x] Upload the well-trained weights for Oei et al.'s model.
