# Large-scale Multi-class Image-based Cell Classification with Deep Learning [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
Tensorflow implementation of "Large-scale Multi-class Image-based Cell Classification with Deep Learning" by Nan M., Edmund Y. L., Tsia K. M., Hayden K-H. So. [[Project]](https://monaen.github.io/Meng2018Largescale)

### Project page
[https://monaen.github.io/Meng2018Largescale](https://monaen.github.io/Meng2018Largescale)

### Code
We provide the training and test code at: [https://github.com/monaen/Meng2018Largescale/tree/code](https://github.com/monaen/Meng2018Largescale/tree/code)

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
```
### TODO
- [x] Update the proposed model -- CellNet
- [x] Upload the small/large dataset
- [x] Upload the augmented dataset
- [x] Add the Hep2 model from [Gao et al.("HEp-2 cell image classification with deep convolutional neural networks")](https://ieeexplore.ieee.org/document/7400923)
- [ ] Add the Hep2 model from [Liu et al.("HEp-2 cell classification based on a Deep Autoencoding-Classification convolutional neural network")](https://ieeexplore.ieee.org/document/7950689)
- [ ] Add the model from [Oei et al.("Convolutional neural network for cell classification using microscope images of intracellular actin networks.")](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0213626)
