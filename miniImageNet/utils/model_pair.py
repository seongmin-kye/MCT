import numpy as np
import random
from random import sample

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.backbone.resnet12_pair import ResNet12


class Runner(object):
    def __init__(self, nb_class_train, nb_class_test, input_size, n_shot, n_query,
                 transductive_train=True, flip=True, drop=True):

        self.nb_class_train = nb_class_train
        self.nb_class_test = nb_class_test
        self.input_size = input_size
        self.n_shot = n_shot
        self.n_query = n_query
        self.is_transductive = transductive_train
        self.flip = flip if transductive_train else False
        self.drop = drop if transductive_train else False

        # create model
        self.model = ResNet12(with_drop=drop)
        self.model.cuda()
        self.CE = nn.CrossEntropyLoss().cuda()
        self.NLL = nn.NLLLoss().cuda()
        self.MSE = nn.MSELoss().cuda()

    def set_optimizer(self, learning_rate, weight_decay_rate, which_optim='SGD'):

        if which_optim == 'SGD':
            self.optimizer = optim.SGD([{'params': self.model.parameters(), 'weight_decay': weight_decay_rate}],
                                       lr=learning_rate, momentum=0.9, nesterov=True)
        elif which_optim == 'Adam':
            self.optimizer = optim.Adam([{'params': self.model.parameters(), 'weight_decay': weight_decay_rate}],
                                        lr=learning_rate)

    def compute_accuracy(self, t_data, prob):
        t_est = torch.argmax(prob, dim=1)

        return (t_est == t_data)

    def make_protomap(self, support_set, nb_class):
        B, C, W, H = support_set.shape
        protomap = support_set.reshape(self.n_shot, nb_class, C, W, H)
        protomap = protomap.mean(dim=0)

        return protomap

    def make_input(self, images):
        images = np.stack(images)
        images = torch.Tensor(images).cuda()
        images = images.view(images.size(0), 84, 84, 3)
        images = images.permute(0, 3, 1, 2)

        return images


    def add_query(self, support_set, query_set, prob, nb_class):

        B, C, W, H = support_set.shape
        per_class = support_set.reshape(self.n_shot, nb_class, C, W, H)

        for i in range(nb_class):
            ith_prob = prob[:, i].reshape(prob.size(0), 1, 1, 1)
            ith_map = torch.cat((per_class[:, i], query_set * ith_prob), dim=0)
            ith_map = torch.sum(ith_map, dim=0, keepdim=True) / (ith_prob.sum() + self.n_shot)
            if i == 0:
                protomap = ith_map
            else:
                protomap = torch.cat((protomap, ith_map), dim=0)

        return protomap

    def flatten(self, set):
        # flatten
        set = torch.flatten(set, start_dim=1)
        set = F.normalize(set)

        return set

    def flip_key(self, images):
        self.model.eval()
        with torch.no_grad():
            flipped_key = self.model(torch.flip(images, dims=[3]))
            return flipped_key


    def train_transduction(self, keys, nb_class):

        key = sample(keys, 1)[0]
        support_set = key[:nb_class * self.n_shot]
        protomap = self.make_protomap(support_set, nb_class)
        query_set = key[nb_class * self.n_shot:]

        # metric scaling
        query_NF = self.flatten(query_set)
        proto_NF = self.flatten(protomap)
        sigma = self.model.relation_net(query_set, protomap)

        diff = query_NF.unsqueeze(1) - proto_NF
        distance = diff.pow(2).sum(dim=2) / sigma ** 2
        prob = F.softmax(-distance, dim=1)

        return prob

    def test_transduction(self, key_o, key_f, nb_class, iters=11):

        key_list = [key_f, key_o] if self.flip else [key_o]
        if not self.is_transductive: iters = 1
        prob_list = []

        for iter in range(iters):
            prob_sum = 0
            for keys in key_list:
                for key in keys:
                    support_set = key[:nb_class * self.n_shot]
                    query_set = key[nb_class * self.n_shot:]

                    # Make Protomap
                    if iter == 0: protomap = self.make_protomap(support_set, nb_class)
                    else: protomap = self.add_query(support_set, query_set,
                                                    prob_list[iter-1], nb_class)

                    # metric scaling
                    query_NF = self.flatten(query_set)
                    proto_NF = self.flatten(protomap)
                    sigma = self.model.relation_net(query_set, protomap)

                    # Calculate distance
                    diff = query_NF.unsqueeze(1) - proto_NF
                    distance = diff.pow(2).sum(dim=2) / sigma ** 2
                    prob = F.softmax(-distance, dim=1)
                    prob_sum += prob / (len(keys) * len(key_list))

            prob_list.append(prob_sum)

        return prob_list[-1]

    def train(self, images, labels):

        nb_class = self.nb_class_train
        labels_DC = torch.tensor(labels, dtype=torch.long).cuda()
        labels_IC = torch.tensor([i for i in range(nb_class)] * (self.n_query), dtype=torch.long).cuda()
        images = self.make_input(images)

        key_f = self.flip_key(images) if (random.random() > 0.5) and (self.flip) else None
        self.model.train()
        key_o = self.model(images)
        key = key_o[0]

        # Dimension-wise classification
        loss_DC = 0
        key_DC = self.model.global_w(key)
        key_DC = key_DC.flatten(start_dim=2)
        for pixel in range(key_DC.size(2)):
            loss_DC += self.CE(key_DC[:,:,pixel], labels_DC) / key_DC.size(2)

        support_set = key[:nb_class * self.n_shot]
        query_set = key[nb_class * self.n_shot:]

        # Make prototype
        if self.is_transductive:
            key_conf = key_o if key_f is None else key_f
            prob = self.train_transduction(key_conf, nb_class)
            protomap = self.add_query(support_set, query_set, prob, nb_class)
        else:
            protomap = self.make_protomap(support_set, nb_class)

        scaled_proto = self.flatten(protomap)
        scaled_query = self.flatten(query_set)
        sigma = self.model.relation_net(query_set, protomap)

        # Instance-wise classification
        diff = scaled_query.unsqueeze(1) - scaled_proto
        distance = diff.pow(2).sum(dim=2) / sigma ** 2
        loss_IC = self.CE(-distance, labels_IC)

        loss = 0.5 * loss_IC + 1 * loss_DC

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data


    def evaluate(self, images, labels):

        nb_class = self.nb_class_test
        images = self.make_input(images)
        labels = torch.tensor(labels, dtype=torch.long).cuda()
        self.model.eval()
        with torch.no_grad():
            key_f = self.model(torch.flip(images, dims=[3])) if self.flip else None
            key_o = self.model(images)

            # ProtoNet
            q_label = labels[nb_class * self.n_shot:]
            prob = self.test_transduction(key_o, key_f, nb_class, iters=11)
            acc = self.compute_accuracy(q_label, prob)

            return acc