## Large-scale Multi-class Image-based Cell Classification with Deep Learning (Tensorflow Implementation)

The tensorflow project for work "Large-scale Multi-class Image-based Cell Classification with Deep Learning"

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
- [ ] Add the Hep2 model from "HEp-2 cell image classification with deep convolutional neural networks"
- [ ] Add the model from "Convolutional neural network for cell classification using microscope images of intracellular actin networks."
