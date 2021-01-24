# Meta-Learned Confidence for Few-shot Learining
Pytorch code for following paper:
* **Title** : Meta-Learned Confidence for Few-shot Learning.[[pdf](https://arxiv.org/abs/2002.12017)]
* **Author** : Seong Min Kye, [Hae Beom Lee](https://haebeom-lee.github.io/), Hoirin Kim, [Sung Ju Hwang](http://www.sungjuhwang.com/)
* **Previous version** : Transductive Few-shot Learning with Meta-Learned Confidence [[paper](https://arxiv.org/abs/2002.12017v1)] / [[codes](https://github.com/seongmin-kye/MCT_DFMN)]
## Abstract
<img align="middle" width="900" src="https://github.com/seongmin-kye/MCT/blob/master/concept_figure.png">
Transductive inference is an effective means of tackling the data deficiency problem in few-shot learning settings. A popular transductive inference technique for few-shot metric-based approaches, is to update the prototype of each class with the mean of the most confident query examples, or confidence-weighted average of all the query samples. However, a caveat here is that the model confidence may be unreliable, which may lead to incorrect predictions. To tackle this issue, we propose to meta-learn the confidence for each query sample, to assign optimal weights to unlabeled queries such that they improve the model's transductive inference performance on unseen tasks. We achieve this by meta-learning an input-adaptive distance metric over a task distribution under various model and data perturbations, which will enforce consistency on the model predictions under diverse uncertainties for unseen tasks. Moreover, we additionally suggest a regularization which explicitly enforces the consistency on the predictions across the different dimensions of a high-dimensional embedding vector. We validate our few-shot learning model with meta-learned confidence on four benchmark datasets, on which it largely outperforms strong recent baselines and obtains new state-of-the-art results. Further application on semi-supervised few-shot learning tasks also yields significant performance improvements over the baselines.

## Requirements
* Python 3.6
* Pytorch 1.3.1

## Data Download
* [**miniImageNet**](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) 
* [**tieredImageNet**](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG)
* [**FC100**](https://drive.google.com/file/d/1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1/view?usp=sharing)
* [**CIFAR-FS**](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)

## Training / Testing with inductive manner
1. 5-way 1-shot / 5-shot
```
# 5-way 1-shot MCI(Instance)
$ python train.py --is_train True --gpu 0 --metric instance --transductive False --flip False --drop False --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 --metric instance --transductive False --flip False --drop False --n_shot 1 --n_train_class 15

# 5-way 5-shot MCI(Instance)
$ python train.py --is_train True --gpu 0 --metric instance --transductive False --flip False --drop False --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 --metric instance --transductive False --flip False --drop False --n_shot 5 --n_train_class 15

# 5-way 5-shot MCI(Pair)
$ python train.py --is_train True --gpu 0 --metric pair --transductive False --flip False --drop False --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 --metric pair --transductive False --flip False --drop False --n_shot 1 --n_train_class 15

# 5-way 5-shot MCI(Pair)
$ python train.py --is_train True --gpu 0 --metric pair --transductive False --flip False --drop False --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 --metric pair --transductive False --flip False --drop False --n_shot 5 --n_train_class 15
```
## Training / Testing with transductive manner
1. 5-way 1-shot / 5-shot
```
# 5-way 1-shot MCI(Instance)
$ python train.py --is_train True --gpu 0 --metric instance --transductive True --flip True --drop True --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 --metric instance --transductive True --flip True --drop True --n_shot 1 --n_train_class 15

# 5-way 5-shot MCI(Instance)
$ python train.py --is_train True --gpu 0 --metric instance --transductive True --flip True --drop True --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 --metric instance --transductive True --flip True --drop True --n_shot 5 --n_train_class 15

# 5-way 5-shot MCI(Pair)
$ python train.py --is_train True --gpu 0 --metric pair --transductive True --flip True --drop True --n_shot 1 --n_train_class 15
$ python train.py --is_train False --gpu 0 --metric pair --transductive True --flip True --drop True --n_shot 1 --n_train_class 15

# 5-way 5-shot MCI(Pair)
$ python train.py --is_train True --gpu 0 --metric pair --transductive True --flip True --drop True --n_shot 5 --n_train_class 15
$ python train.py --is_train False --gpu 0 --metric pair --transductive True --flip True --drop True --n_shot 5 --n_train_class 15
```

## Acknowledgments

This code is based on the implementation of [**TapNet**](https://github.com/istarjun/TapNet), [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet). And we use the dataset from [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet).



