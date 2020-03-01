# Transductive Few-shot Learning with Meta-Learned Confidence
Pytorch code for following paper,
* **Title** : [**Transductive Few-shot Learning with Meta-Learned Confidence.**](https://arxiv.org/abs/2002.12017)
* **Author** : Seong Min Kye, Hae Beom Lee, Hoirin Kim, Sung Ju Hwang

## Requirements
* Python 3.6
* Pytorch 1.3.1

## Data Download
* [**miniImageNet**](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) 
* [**tieredImageNet**](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG)
* [**FC100**](https://drive.google.com/file/d/1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1/view?usp=sharing)
* [**CIFAR-FS**](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)

## Training/Testing with inductive manner
1. miniImageNet 5-way 1-shot/5-shot
```
# miniImageNet, 5-way 1-shot
$ python train.py --is_train True --gpu 0 Transductive False --flip False --drop False --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 Transductive False --flip False --drop False --n_shot 1 --n_train_class 15

# miniImageNet, 5-way 5-shot
$ python train.py --is_train True --gpu 0 Transductive False --flip False --drop False --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 Transductive False --flip False --drop False --n_shot 5 --n_train_class 15
```
2 tieredImageNet 5-way 1-shot/5-shot
```
# tieredImageNet, 5-way 1-shot
$ python train.py --is_train True --gpu 0 Transductive False --flip False --drop False --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 Transductive False --flip False --drop False --n_shot 1 --n_train_class 15

# tieredImageNet, 5-way 5-shot
$ python train.py --is_train True --gpu 0 Transductive False --flip False --drop False --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 Transductive False --flip False --drop False --n_shot 5 --n_train_class 15
```
## Training/Testing with transductive manner
1. miniImageNet 5-way 1-shot/5-shot
```
# miniImageNet, 5-way 1-shot
$ python train.py --is_train True --gpu 0 Transductive True --flip True --drop True --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 Transductive True --flip True --drop True --n_shot 1 --n_train_class 15

# miniImageNet, 5-way 5-shot
$ python train.py --is_train True --gpu 0 Transductive True --flip True --drop True --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 Transductive True --flip True --drop True --n_shot 5 --n_train_class 15
```
2. tieredImageNet 5-way 1-shot/5-shot
```
# tieredImageNet, 5-way 1-shot
$ python train.py --is_train True --gpu 0 Transductive True --flip True --drop True --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 Transductive True --flip True --drop True --n_shot 1 --n_train_class 15

# tieredImageNet, 5-way 5-shot
$ python train.py --is_train True --gpu 0 Transductive True --flip True --drop True --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 Transductive True --flip True --drop True --n_shot 5 --n_train_class 15
```

## Acknowledgments

This code is based on the implementation of [**TapNet**](https://github.com/istarjun/TapNet), [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet). And we use the dataset from [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet).



