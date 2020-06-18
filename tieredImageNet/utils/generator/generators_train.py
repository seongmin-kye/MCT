"""
This code based on codes from https://github.com/tristandeleu/ntm-one-shot \
                              and https://github.com/kjunelee/MetaOptNet
"""
import numpy as np
import random
import pickle as pkl
from PIL import Image
from torchvision import transforms
from utils.generator.randaugment import RandAugmentMC, CutoutAbs

class tieredImageNetGenerator(object):

    def __init__(self, data_file, label_file, nb_classes=5, n_shot=5, n_query=8,
                  max_iter=None, xp=np):
        super(tieredImageNetGenerator, self).__init__()
        self.data_file = data_file
        self.label_file = label_file
        self.nb_classes = nb_classes
        self.n_shot = n_shot
        self.n_query = n_query
        self.nb_samples_per_class = n_shot+n_query
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.data = self._load_data()
        self.transform = transforms.Compose([RandAugmentMC(n=2, m=10)])

    def _load_data(self):
        data = np.load(self.data_file)['images']
        labels = self.load_data(self.label_file)['labels']
        label2ind = self.buildLabelIndex(labels)

        return {key: np.array(data[val]) for (key, val) in label2ind.items()}

    def load_data(self, data_file):
        try:
            with open(data_file, 'rb') as fo:
                data = pkl.load(fo)
            return data
        except:
            with open(data_file, 'rb') as f:
                u = pkl._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data

    def buildLabelIndex(self, labels):
        label2inds = {}
        for idx, label in enumerate(labels):
            if label not in label2inds:
                label2inds[label] = []
            label2inds[label].append(idx)

        return label2inds


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, labels = self.sample(self.nb_classes, self.nb_samples_per_class)

            return (self.num_iter - 1), (images, labels)
        else:
            raise StopIteration()


    def weak_aug(self, img):

        # random flipping
        if random.random() > 0.5:
            img = np.flip(img, 1)

        # random shifting
        npad = ((8, 8), (8, 8), (0, 0))
        img = np.pad(img, npad, 'constant', constant_values=(127.0))
        x = random.randint(0, 16)
        y = random.randint(0, 16)
        img = img[y:y + 84, x:x + 84]

        return img

    def strong_aug(self, img):

        # random flipping
        if random.random() > 0.5:
            img = np.flip(img, 1)

        # random shifting
        npad = ((8, 8), (8, 8), (0, 0))
        img = np.pad(img, npad, 'constant', constant_values=(127.0))
        x = random.randint(0, 16)
        y = random.randint(0, 16)
        img = img[y:y + 84, x:x + 84]

        img = Image.fromarray(img, 'RGB')
        img = self.transform(img)
        img = CutoutAbs(img, 42)
        img = np.array(img)

        return img

    def sample(self, nb_classes, nb_samples_per_class):

        picture_list = sorted(set(self.data.keys()))
        pic_to_idx = {pic: i for i, pic in enumerate(picture_list)}
        sampled_characters = random.sample(self.data.keys(), nb_classes)

        labels_and_images = []
        for (k, char) in enumerate(sampled_characters):
            label = pic_to_idx[char]
            _imgs = self.data[char]
            _ind = random.sample(range(len(_imgs)), nb_samples_per_class)
            labels_and_images.extend(
                [(label, self.xp.array(self.weak_aug(_imgs[i]) / np.float32(255))) for i in _ind[:self.n_shot]])
            labels_and_images.extend(
                [(label, self.xp.array(self.strong_aug(_imgs[i]) / np.float32(255))) for i in _ind[self.n_shot:]])

        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_labels_and_images.extend([labels_and_images[i + j * self.nb_samples_per_class]])

        labels, images = zip(*arg_labels_and_images)

        return images, labels