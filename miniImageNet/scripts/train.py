import os
import sys
sys.path.append('../')
import argparse
import numpy as np

import torch
from utils.generator.generators_train import miniImageNetGenerator as train_loader
from utils.generator.generators_test import miniImageNetGenerator as test_loader

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type=str2bool, default=True,
                        help='Choice train or test.')
    parser.add_argument('--n_folder', type=int, default=0,
                        help='Number of folder.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number.')

    max_iter = 50000
    parser.add_argument('--use_checkpoint', type=str2bool, default=False,
                        help='Use checkpoint.')
    parser.add_argument('--optim', type=str, default='SGD',
                        help='Choice optimizer.')
    parser.add_argument('--first_decay', type=int, default=25000,
                        help='First decay step.')
    parser.add_argument('--second_decay', type=int, default=35000,
                        help='Second decay step.')

    parser.add_argument('--metric', type=str, default='instance',
                        help='Choice the metric such as instance and pair.')
    parser.add_argument('--transductive', type=str2bool, default=True,
                        help='Whether to use transductive training or not.')
    parser.add_argument('--flip', type=str2bool, default=True,
                        help='Whether to inject data uncertainty.')
    parser.add_argument('--drop', type=str2bool, default=True,
                        help='Whether to inject model uncertainty.')

    parser.add_argument('--n_shot', type=int, default=5,
                        help='Number of support set per class in train.')
    parser.add_argument('--n_query', type=int, default=8,
                        help='Number of queries per class in train.')
    parser.add_argument('--n_test_query', type=int, default=15,
                        help='Number of queries per class in test.')
    parser.add_argument('--n_train_class', type=int, default=15,
                        help='Number of way for training episode.')
    parser.add_argument('--n_test_class', type=int, default=5,
                        help='Number of way for test episode.')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    #######################
    folder_num = args.n_folder

    # optimizer setting
    lrstep2 = args.second_decay
    lrstep1 = args.first_decay
    initial_lr = 1e-1 if args.optim == 'SGD' else 1e-3

    # train episode setting
    n_shot=args.n_shot
    n_query=args.n_query
    nb_class_train = args.n_train_class

    # test episode setting
    n_query_test = args.n_test_query
    nb_class_test=args.n_test_class

    # You can download dataset from https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view
    #data path
    data_path = '/data/miniImageNet'
    train_path = data_path + '/miniImageNet_category_split_train_phase_train.pickle'
    val_path = data_path + '/miniImageNet_category_split_val.pickle'
    test_path = data_path + '/miniImageNet_category_split_test.pickle'

    #save_path
    save_path = 'save/baseline_' + str(folder_num).zfill(3)
    filename_5shot=save_path + '/miniImageNet_ResNet12'
    filename_5shot_last= save_path + '/miniImageNet_ResNet12_last'

    # set up training
    # ------------------
    if args.metric == 'instance':
        from utils.model_instance import Runner
    elif args.metric == 'pair':
        from utils.model_pair import Runner

    model = Runner(nb_class_train=nb_class_train, nb_class_test=nb_class_test, input_size=3*84*84,
                   n_shot=n_shot, n_query=n_query,
                   transductive_train=args.transductive, flip=args.flip, drop=args.drop)
    model.set_optimizer(learning_rate=initial_lr, weight_decay_rate=5e-4, which_optim=args.optim)

    start_episode = 0
    if args.use_checkpoint:
        checkpoint = torch.load(filename_5shot_last)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_h=[]
    accuracy_h_val=[]
    accuracy_h_test=[]

    acc_best=0
    epoch_best=0
    # start training
    # ----------------
    if args.is_train:
        train_generator = train_loader(data_file=train_path, nb_classes=nb_class_train,
                                       n_shot=n_shot, n_query=n_query, max_iter=max_iter-start_episode)

        for t, (images, labels) in train_generator:

            if args.use_checkpoint:
                t += start_episode

            # train
            loss = model.train(images, labels)
            # logging
            loss_h.extend([loss.tolist()])
            if (t % 100 == 0):
                print("Episode: %d, Train Loss: %f "%(t, loss))
                torch.save({
                    'episode': t,
                    'model_state_dict': model.model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'loss': loss,
                }, filename_5shot_last)

            if (t != 0) and (t % 1000 == 0):
                print('Evaluation in Validation data')
                test_generator = test_loader(data_file=val_path, nb_classes=nb_class_test,
                                             nb_samples_per_class=n_shot+n_query_test, max_iter=600)
                scores = []
                for i, (images, labels) in test_generator:
                    acc = model.evaluate(images, labels)
                    score = acc.data.cpu().numpy()
                    scores.append(score)

                print(('Accuracy {}-shot ={:.2f}%').format(n_shot, 100*np.mean(np.array(scores))))
                accuracy_t=100*np.mean(np.array(scores))

                if acc_best < accuracy_t:
                    acc_best = accuracy_t
                    epoch_best=t
                    torch.save({
                        'episode': t,
                        'model_state_dict': model.model.state_dict(),
                        'optimizer_state_dict': model.optimizer.state_dict(),
                        'loss': loss,
                    }, filename_5shot)
                accuracy_h_val.extend([accuracy_t.tolist()])
                del(test_generator)
                del(acc)
                del(accuracy_t)

                print('Evaluation in Test data')
                test_generator = test_loader(data_file=test_path, nb_classes=nb_class_test,
                                             nb_samples_per_class=n_shot+n_query_test, max_iter=600)
                scores = []
                for i, (images, labels) in test_generator:
                    acc = model.evaluate(images, labels)
                    score = acc.data.cpu().numpy()
                    scores.append(score)

                print(('Accuracy {}-shot ={:.2f}%').format(n_shot, 100*np.mean(np.array(scores))))
                accuracy_t=100*np.mean(np.array(scores))

                accuracy_h_test.extend([accuracy_t.tolist()])
                del(test_generator)
                del(acc)
                del(accuracy_t)
                if len(accuracy_h_val) > 5:
                    print('***Average accuracy on past 5 test acc***')
                    print('Best epoch =',epoch_best,'Best {}-shot acc='.format(n_shot), acc_best)

            decay1 = 0.06 if args.optim == 'SGD' else 0.1
            decay2 = 0.2 if args.optim == 'SGD' else 0.1
            if (t != 0) & (t == lrstep1):
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] *= decay1
                    print('-------Decay Learning Rate to ', param_group['lr'], '------')
            if (t != 0) & (t == lrstep2):
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] *= decay2
                    print('-------Decay Learning Rate to ', param_group['lr'], '------')

    accuracy_h5=[]
    total_acc = []
    model.model.load_state_dict(torch.load(filename_5shot)['model_state_dict'])
    print('Evaluating the best {}-shot model...'.format(n_shot))
    for i in range(10):
        test_generator = test_loader(data_file=test_path, nb_classes=nb_class_test,
                                     nb_samples_per_class=n_shot+n_query_test, max_iter=100)
        scores=[]
        for j, (images, labels) in test_generator:
            acc = model.evaluate(images, labels)
            score = acc.data.cpu().numpy()
            scores.append(score)
            total_acc.append(np.mean(score) * 100)

        accuracy_t=100*np.mean(np.array(scores))
        accuracy_h5.extend([accuracy_t.tolist()])
        print(('100 episodes with 15-query accuracy: {}-shot ={:.2f}%').format(n_shot, accuracy_t))
        del(test_generator)
        del(acc)
        del(accuracy_t)

    stds = np.std(total_acc, axis=0)
    ci95 = 1.96 * stds / np.sqrt(len(total_acc))
    print(('Accuracy_test {}-shot ={:.2f}({:.2f})').format(n_shot, np.mean(accuracy_h5), ci95))
