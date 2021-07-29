# Large-Scale Multi-Class Image-Based Cell Classification with Deep Learning [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
Tensorflow implementation of "Large-scale Multi-class Image-based Cell Classification with Deep Learning" by Nan M., Edmund Y. L., Tsia K. M., Hayden K-H. So. [[Paper]](https://monaen.github.io/Large-Scale-Multi-Class-Image-Based-Cell-Classification-with-Deep-Learning/materials/Largescale_Multiclass_Imagebased_Cell_Classification_with_Deep_Learning.pdf)


### Project page
[https://monaen.github.io/Large-Scale-Multi-Class-Image-Based-Cell-Classification-with-Deep-Learning
](https://monaen.github.io/Large-Scale-Multi-Class-Image-Based-Cell-Classification-with-Deep-Learning
)

### Environment
* Ubuntu 14.04/16.04
* Python 2.7
* Note: for Windows Users you need to download the dataset and unzip to the "data" folder.

### Code
We provide the training and test code at: [https://github.com/monaen/Large-Scale-Multi-Class-Image-Based-Cell-Classification-with-Deep-Learning/tree/code](https://github.com/monaen/Meng2018Largescale/tree/code)

### Dataset
We provide three different datasets, and all the datasets could be downloaded at [https://github.com/monaen/Large-Scale-Multi-Class-Image-Based-Cell-Classification-with-Deep-Learning/tree/data](https://github.com/monaen/Meng2018Largescale/tree/data)

* download the small data
```commandline
git clone https://github.com/monaen/Large-Scale-Multi-Class-Image-Based-Cell-Classification-with-Deep-Learning
.git --branch code --single-branch
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

* run the demo
```commandline
python demo.py
```

### Details
The entire [code project](https://github.com/monaen/Large-Scale-Multi-Class-Image-Based-Cell-Classification-with-Deep-Learning/tree/code) contains many recent deep learning models for cell classification. We implement these models according to their papers and integrate them into our project.
* cellnet: our work
* gaohep2net: [Gao et al.("HEp-2 cell image classification with deep convolutional neural networks")](https://ieeexplore.ieee.org/document/7400923)
* liuhep2net: [Liu et al.("HEp-2 cell classification based on a Deep Autoencoding-Classification convolutional neural network")](https://ieeexplore.ieee.org/document/7950689)
* oeinet: [Oei et al.("Convolutional neural network for cell classification using microscope images of intracellular actin networks.")](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0213626)


**If you want to use our data or code please cite our paper properly.**

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
