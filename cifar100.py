# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from PIL import Image
import os
import os.path
import sys


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import numpy as np

import torch
from torchvision import datasets, transforms
import random
from utils import *
from util import *


'''
g_y: global label,
y: local label for training,
tt: private module id
td: discriminator label

'''
class iCIFAR10(datasets.CIFAR100):

    def __init__(self, root, classes, memory_classes, memory, task_num, train, transform=None, target_transform=None, download=True):

        super(iCIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform, download=True)
        self.train = train  # training set or test set
        if not isinstance(classes, list):
            classes = [classes]
        self.task_num = task_num
        self.class_mapping = {c: i for i, c in enumerate(classes)}
        self.class_indices = {}
        self.memory_classes = memory_classes
        self.memory = memory
        for cls in classes:
            self.class_indices[self.class_mapping[cls]] = []

        if self.train:
            train_data = []
            train_global_labels = []
            train_labels = []
            train_tt = []  # task module labels
            train_td = []  # disctiminator labels

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_global_labels.append(self.targets[i])
                    train_labels.append(self.class_mapping[self.targets[i]])
                    train_tt.append(task_num)
                    train_td.append(task_num+1)
                    self.class_indices[self.class_mapping[self.targets[i]]].append(i)



            self.train_data = train_data
            self.train_global_labels = train_global_labels
            self.train_labels = train_labels
            self.train_tt = train_tt
            self.train_td = train_td


        if not self.train:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:

                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

            test_data = []
            test_global_labels = []
            test_labels = []
            test_tt = []  # task module labels
            test_td = []  # disctiminator labels
            for i in range(len(self.test_data)):
                if self.test_labels[i] in classes:
                    test_data.append(self.test_data[i])
                    test_global_labels.append(self.test_labels[i])
                    test_labels.append(self.class_mapping[self.test_labels[i]])
                    test_tt.append(task_num)
                    test_td.append(task_num + 1)
                    self.class_indices[self.class_mapping[self.test_labels[i]]].append(i)

            self.test_data = np.array(test_data)
            self.test_global_labels = test_global_labels
            self.test_labels = test_labels
            self.test_tt = test_tt
            self.test_td = test_td

    def update_memory(self):
        if self.memory_classes:
            for task_id in range(self.task_num):
                for j in self.memory_classes[task_id]:
                    len_memory = len(self.memory[task_id][j]['x'])
                    if len_memory == 0:
                        pass
                    else:
                        for k in range(len_memory):
                            self.train_data.append(self.memory[task_id][j]['x'][k])
                            self.train_global_labels.append(self.memory[task_id][j]['g_y'][k])
                            self.train_labels.append(self.memory[task_id][j]['y'][k])
                            self.train_tt.append(self.memory[task_id][j]['tt'][k])
                            self.train_td.append(self.memory[task_id][j]['td'][k])

    def __getitem__(self, index):
        if self.train:
            img, g_target, target, tt, td = self.train_data[index], self.train_global_labels[index], self.train_labels[index], self.train_tt[index], self.train_td[index]
        else:
            img, g_target, target, tt, td = self.test_data[index], self.test_global_labels[index], self.test_labels[index], self.test_tt[index], self.test_td[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img)
        except:
            pass

        try:
            if self.transform is not None:
                img = self.transform(img)
        except:
            pass
        try:
            if self.target_transform is not None:
                g_target = self.target_transform(g_target)
                target = self.target_transform(target)
        except:
            pass

        return img, g_target, target, tt, td




    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }



class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, args):
        super(DatasetGen, self).__init__()

        self.seed = args.seed
        self.batch_size=args.batch_size
        self.pc_valid=args.pc_valid
        self.root = args.data_folder
        # self.latent_dim = args.latent_dim

        self.num_tasks = args.num_tasks
        self.num_classes = args.n_cls

        self.num_samples = args.mem_size


        self.inputsize = [3,32,32]
        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]

        # self.transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        normalize = transforms.Normalize(mean=mean, std=std)
        train_transform = transforms.Compose([
            transforms.Resize(size=(args.size, args.size)),
            # transforms.RandomResizedCrop(size=args.size, scale=(0.1 if args.dataset == 'tiny-imagenet' else 0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=args.size // 20 * 2 + 1, sigma=(0.1, 2.0))],
            #                        p=0.5 if args.size > 32 else 0.0),
            transforms.ToTensor(),
            normalize,
        ])
        self.transformation = TwoCropTransform(train_transform)  # different transform results to build positive pairs

        test_transform = transforms.Compose([
            transforms.Resize(size=(args.size,args.size)),
            transforms.ToTensor(),
            normalize,
        ])


        self.test_transformation = test_transform

        self.taskcla = [[t, int(self.num_classes/self.num_tasks)] for t in range(self.num_tasks)]

        self.indices = {}
        self.dataloaders = {}
        self.idx={}

        self.num_workers = args.workers
        self.pin_memory = True

        np.random.seed(self.seed)
        # task_ids = np.split(np.random.permutation(self.num_classes),self.num_tasks)
        task_ids = np.split(np.array([i for i in range(self.num_classes)]),self.num_tasks)

        self.task_ids = [list(arr) for arr in task_ids]


        self.train_set = {}
        self.test_set = {}
        self.train_split = {}

        self.task_memory = {}

        for i in range(self.num_tasks):  # saved samples for replay
            self.task_memory[i] = {}
            for j in self.task_ids[i]:
                self.task_memory[i][j] = {}
                self.task_memory[i][j]['x'] = []
                self.task_memory[i][j]['g_y'] = []  # global label
                self.task_memory[i][j]['y'] = []  # local task label
                self.task_memory[i][j]['tt'] = []  # task id
                self.task_memory[i][j]['td'] = []  #

        self.use_memory = args.use_memory

    def get(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()


        if task_id == 0:
            memory_classes = None
            memory=None
        else:
            memory_classes = self.task_ids
            memory = self.task_memory

        self.train_set[task_id] = iCIFAR10(root=self.root, classes=self.task_ids[task_id], memory_classes=memory_classes,
                                            memory=memory, task_num=task_id, train=True, download=True, transform=self.transformation)
        self.test_set[task_id] = iCIFAR10(root=self.root, classes=self.task_ids[task_id], memory_classes=None,
                                           memory=None, task_num=task_id, train=False,
                                     download=True, transform=self.transformation)




        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id],
                                                                 [len(self.train_set[task_id]) - split, split])
        if self.use_memory:
            train_split.dataset.update_memory()
            reply_indices = np.where(np.array(train_split.dataset.train_tt) < task_id)[0].tolist()
            train_split.indices = train_split.indices + reply_indices
        # self.train_split[task_id] = train_split

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory, shuffle=True)
        # np.savez('./data/task_0_data', train_loader.dataset.dataset.train_data, train_loader.dataset.dataset.train_labels)
        # torch.save(train_loader, './data/memory/dataloader.pth')

        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=self.batch_size,
                                                   num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        # test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
        #                                           pin_memory=self.pin_memory,shuffle=True)


        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        # self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = 'CIFAR100-{}-{}'.format(task_id,self.task_ids[task_id])

        print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Validation set size: {} images of {}x{}".format(len(valid_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Train+Val  set size: {} images of {}x{}".format(len(valid_loader.dataset)+len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        # print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        if self.use_memory and self.num_samples > 0:
            self.update_memory(task_id)

        return self.dataloaders

    def get_test_dataloader(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        self.test_set[task_id] = iCIFAR10(root=self.root, classes=self.task_ids[task_id], memory_classes=None,
                                           memory=None, task_num=task_id, train=False,
                                     download=True, transform=self.transformation)

        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                  pin_memory=self.pin_memory,shuffle=True)

        self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = 'CIFAR100-{}-{}'.format(task_id,self.task_ids[task_id])
        print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        return self.dataloaders

    def update_memory(self, task_id):


        num_classes_per_task = int(self.num_classes/self.num_tasks)
        # num_samples_per_class = self.num_samples // (int(self.num_classes/self.num_tasks) * (task_id+1))
        num_samples_per_class_previous = math.floor(self.num_samples / (num_classes_per_task * (task_id + 1)))
        num_samples_per_class_new = math.ceil((self.num_samples - (num_classes_per_task * task_id) * num_samples_per_class_previous) / num_classes_per_task)
        mem_class_mapping = {c: i for i, c in enumerate(self.task_ids[task_id])}

        # Looping over each class in the current task
        for class_id in self.task_ids[task_id]:
            p = np.where(np.array(self.train_set[task_id].train_global_labels == class_id))[0].tolist()
            s = data.Subset(self.train_set[task_id], p)
            data_loader = data.DataLoader(s, batch_size=1, num_workers=self.num_workers, pin_memory=self.pin_memory,
                                          shuffle=True)
            # data_loader = torch.utils.data.DataLoader(self.train_set[task_id], batch_size=1,
            #                                             num_workers=self.num_workers,
            #                                             pin_memory=self.pin_memory)

            randind = torch.randperm(len(data_loader.dataset))[:num_samples_per_class_new]  # randomly sample some data

            for ind in randind:
                self.task_memory[task_id][class_id]['x'].append(data_loader.dataset[ind][0])
                self.task_memory[task_id][class_id]['g_y'].append(data_loader.dataset[ind][1])
                self.task_memory[task_id][class_id]['y'].append(mem_class_mapping[class_id])
                self.task_memory[task_id][class_id]['tt'].append(data_loader.dataset[ind][3])
                self.task_memory[task_id][class_id]['td'].append(data_loader.dataset[ind][4])

        for i in range(task_id):
            for j in range(num_classes_per_task):
                self.task_memory[i][self.task_ids[i][j]]['x'] = self.task_memory[i][self.task_ids[i][j]]['x'][
                                                                :num_samples_per_class_previous]
                self.task_memory[i][self.task_ids[i][j]]['g_y'] = self.task_memory[i][self.task_ids[i][j]]['g_y'][
                                                                  :num_samples_per_class_previous]
                self.task_memory[i][self.task_ids[i][j]]['y'] = self.task_memory[i][self.task_ids[i][j]]['y'][
                                                                :num_samples_per_class_previous]
                self.task_memory[i][self.task_ids[i][j]]['tt'] = self.task_memory[i][self.task_ids[i][j]]['tt'][
                                                                 :num_samples_per_class_previous]
                self.task_memory[i][self.task_ids[i][j]]['td'] = self.task_memory[i][self.task_ids[i][j]]['td'][
                                                                 :num_samples_per_class_previous]

        print('Memory updated.')

    # def update_memory(self, task_id):
    #     if not os.path.exists('./data/memory'):
    #         os.mkdir('./data/memory')
    #     if os.path.exists('./data/memory/task_{}_memory.npz'.format(task_id)):
    #         memory = np.load('./data/memory/task_{}_memory.npz'.format(task_id), allow_pickle=True)
    #         self.task_memory[task_id]['x'].append(np.array(memory['x']))
    #         self.task_memory[task_id]['g_y'].append(memory['g_y'])
    #         self.task_memory[task_id]['y'].append(memory['y'])
    #         self.task_memory[task_id]['tt'].append(memory['tt'])
    #         self.task_memory[task_id]['td'].append(memory['td'])
    #     else:
    #         num_samples_per_class = self.num_samples // len(self.task_ids[task_id])
    #         mem_class_mapping = {i: i for i, c in enumerate(self.task_ids[task_id])}
    #
    #
    #         # Looping over each class in the current task
    #         for i in range(len(self.task_ids[task_id])):
    #             # Getting all samples for this class
    #             data_loader = torch.utils.data.DataLoader(self.train_split[task_id], batch_size=1,
    #                                                         num_workers=self.num_workers,
    #                                                         pin_memory=self.pin_memory)
    #             # Randomly choosing num_samples_per_class for this class
    #             randind = torch.randperm(len(data_loader.dataset))[:num_samples_per_class]
    #
    #             # Adding the selected samples to memory
    #             for ind in randind:
    #                 self.task_memory[task_id]['x'].append(data_loader.dataset[ind][0])
    #                 self.task_memory[task_id]['g_y'].append(data_loader.dataset[ind][1])
    #                 self.task_memory[task_id]['y'].append(mem_class_mapping[i])
    #                 self.task_memory[task_id]['tt'].append(data_loader.dataset[ind][3])
    #                 self.task_memory[task_id]['td'].append(data_loader.dataset[ind][4])
    #
    #         print ('Memory updated by adding {} images'.format(len(self.task_memory[task_id]['x'])))
    #         np.savez('./data/memory/task_{}_memory.npz'.format(task_id), x=np.array(self.task_memory[task_id]['x']), g_y=self.task_memory[task_id]['g_y'], y=self.task_memory[task_id]['y'], tt=self.task_memory[task_id]['tt'], td=self.task_memory[task_id]['td'])