# Transductive Few-shot Learning with Meta-Learned Confidence
Pytorch code for 
**Transductive Few-shot Learning with Meta-Learned Confidence.**[pdf](https://arxiv.org/abs/2002.12017)
Seong Min Kye, Hae Beom Lee, Hoirin Kim, Sung Ju Hwang

## Requirements
* Python 3.6
* Pytorch 1.3.1

## Data Download
* [**miniImageNet**](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) 
* [**tieredImageNet**](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG)
* [**FC100**](https://drive.google.com/file/d/1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1/view?usp=sharing)
* [**CIFAR-FS**](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)

For example, in /mini_ImageNet/scripts/train.py line 72:
    ```python
    data_path = 'data/path/miniImageNet'
    ```
    
    ```python
    _MINI_IMAGENET_DATASET_DIR = 'path/to/miniImageNet'
    ```


