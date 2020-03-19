# Transductive Few-shot Learning with Meta-Learned Confidence
Pytorch code for following paper:
* **Title** : [**Transductive Few-shot Learning with Meta-Learned Confidence.**](https://arxiv.org/abs/2002.12017)
* **Author** : Seong Min Kye, [Hae Beom Lee](https://github.com/haebeom-lee), Hoirin Kim, Sung Ju Hwang

## Abstract
<img align="right" width="400" src="https://github.com/seongmin-kye/MCT_DFMN/blob/master/concept.PNG">
We propose a novel transductive inference framework for metric-based meta-learning models, which updates the prototype of each class with the confidence-weighted average of all the support and query samples. However, a caveat here is that the model confidence may be unreliable, which could lead to incorrect prediction in the transductive setting. To tackle this issue, we further propose to meta-learn to assign correct confidence scores to unlabeled queries. Specifically, we meta-learn the parameters of the distance-metric, such that the model can improve its transductive inference performance on unseen tasks with the generated confidence scores. We also consider various types of uncertainties to further enhance the reliability of the meta-learned confidence. We combine our transductive meta-learning scheme, Meta-Confidence Transduction (MCT) with a novel dense classifier, Dense Feature Matching Network (DFMN), which performs both instance-level and feature-level classification without global average pooling and validate it on four benchmark datasets. Our model achieves state-of-the-art results on all datasets, outperforming existing state-of-the-art models by 11.11% and 7.68% on miniImageNet and tieredImageNet dataset respectively. Further qualitative analysis confirms that this impressive performance gain is indeed due to its ability to assign high confidence to instances with the correct labels.

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
$ python train.py --is_train True --gpu 0 --transductive False --flip False --drop False --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 --transductive False --flip False --drop False --n_shot 1 --n_train_class 15

# miniImageNet, 5-way 5-shot
$ python train.py --is_train True --gpu 0 --transductive False --flip False --drop False --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 --transductive False --flip False --drop False --n_shot 5 --n_train_class 15
```
2 tieredImageNet 5-way 1-shot/5-shot
```
# tieredImageNet, 5-way 1-shot
$ python train.py --is_train True --gpu 0 --transductive False --flip False --drop False --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 --transductive False --flip False --drop False --n_shot 1 --n_train_class 15

# tieredImageNet, 5-way 5-shot
$ python train.py --is_train True --gpu 0 --transductive False --flip False --drop False --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 --transductive False --flip False --drop False --n_shot 5 --n_train_class 15
```
## Training/Testing with transductive manner
1. miniImageNet 5-way 1-shot/5-shot
```
# miniImageNet, 5-way 1-shot
$ python train.py --is_train True --gpu 0 --transductive True --flip True --drop True --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 --transductive True --flip True --drop True --n_shot 1 --n_train_class 15

# miniImageNet, 5-way 5-shot
$ python train.py --is_train True --gpu 0 --transductive True --flip True --drop True --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 --transductive True --flip True --drop True --n_shot 5 --n_train_class 15
```
2. tieredImageNet 5-way 1-shot/5-shot
```
# tieredImageNet, 5-way 1-shot
$ python train.py --is_train True --gpu 0 --transductive True --flip True --drop True --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 --transductive True --flip True --drop True --n_shot 1 --n_train_class 15

# tieredImageNet, 5-way 5-shot
$ python train.py --is_train True --gpu 0 --transductive True --flip True --drop True --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 --transductive True --flip True --drop True --n_shot 5 --n_train_class 15
```

## Acknowledgments

This code is based on the implementation of [**TapNet**](https://github.com/istarjun/TapNet), [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet). And we use the dataset from [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet).



