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
from torchvision import transforms
import torchvision.datasets as datasets
from utils import *
from util import *
import PIL


'''
The original source code can be found in
https://github.com/aimagelab/mammoth/blob/master/datasets/seq_tinyimagenet.py
'''


class TinyImagenet(data.Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from google_drive_downloader import GoogleDriveDownloader as gdd

                # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
                print('Downloading dataset')
                gdd.download_file_from_google_drive(
                    file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',

                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                    unzip=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'tiny-imagenet-processed/processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.labels = []
        for num in range(20):
            self.labels.append(np.load(os.path.join(
                root, 'tiny-imagenet-processed/processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.labels = np.concatenate(np.array(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class iTinyImagenet(TinyImagenet):

    def __init__(self, root, classes, memory_classes, memory, task_num, train, transform=None):
        super(iTinyImagenet, self).__init__(root=root, train=train)

        self.transform = transform
        self.train = train
        if not isinstance(classes, list):
            classes = [classes]

        self.class_mapping = {c: i for i, c in enumerate(classes)}
        self.class_indices = {}

        for cls in classes:
            self.class_indices[self.class_mapping[cls]] = []

        self.memory_classes = memory_classes
        self.memory = memory
        self.task_num =task_num
        # data = []
        # labels = []
        # tt = []  # task module labels
        # td = []  # disctiminator labels
        #
        # for i in range(len(self.data)):
        #     if self.labels[i] in classes:
        #         data.append(self.data[i])
        #         labels.append(self.class_mapping[self.labels[i]])
        #         tt.append(task_num)
        #         td.append(task_num+1)
        #         self.class_indices[self.class_mapping[self.labels[i]]].append(i)
        #
        # if memory_classes:
        #     for task_id in range(task_num):
        #         for i in range(len(memory[task_id]['x'])):
        #             if memory[task_id]['y'][i] in range(len(memory_classes[task_id])):
        #                 data.append(memory[task_id]['x'][i])
        #                 labels.append(memory[task_id]['y'][i])
        #                 tt.append(memory[task_id]['tt'][i])
        #                 td.append(memory[task_id]['td'][i])
        #
        # self.data = np.array(data)
        # self.labels = labels
        # self.tt = tt
        # self.td = td

        # if self.train:
        train_data = []
        train_global_labels = []
        train_labels = []
        train_tt = []  # task module labels
        train_td = []  # disctiminator labels

        for i in range(len(self.data)):
            if self.labels[i] in classes:
                train_data.append(self.data[i])
                train_global_labels.append(self.labels[i])
                train_labels.append(self.class_mapping[self.labels[i]])
                train_tt.append(task_num)
                train_td.append(task_num + 1)
                self.class_indices[self.class_mapping[self.labels[i]]].append(i)

        # if self.memory_classes:
        #     for task_id in range(self.task_num):
        #         for j in self.memory_classes[task_id]:
        #             len_memory = len(self.memory[task_id][j]['x'])
        #             if len_memory == 0:
        #                 pass
        #             else:
        #                 for k in range(len_memory):
        #                     train_data.append(self.memory[task_id][j]['x'][k])
        #                     train_global_labels.append(self.memory[task_id][j]['g_y'][k])
        #                     train_labels.append(self.memory[task_id][j]['y'][k])
        #                     train_tt.append(self.memory[task_id][j]['tt'][k])
        #                     train_td.append(self.memory[task_id][j]['td'][k])


        self.train_data = train_data
        self.train_global_labels = train_global_labels
        self.train_labels = train_labels
        self.train_tt = train_tt
        self.train_td = train_td

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
        # img, target, tt, td = self.data[index], self.labels[index], self.tt[index], self.td[index]
        img, g_target, target, tt, td = self.train_data[index], self.train_global_labels[index], self.train_labels[
            index], self.train_tt[index], self.train_td[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            if isinstance(img, list):
                img = img
            else:
                img = Image.fromarray(np.uint8(255 * img))
                if self.transform is not None:
                    img = self.transform(img)
        except:
            pass
        # try:
        #     if self.transform is not None:
        #         img = self.transform(img)
        # except:
        #     pass
        # if not torch.is_tensor(img):
        #     img = Image.fromarray(img)
        #     img = self.transform(img)
        return img, g_target, target, tt, td




    def __len__(self):

        return len(self.train_data)





class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, args):
        super(DatasetGen, self).__init__()

        self.seed = args.seed
        self.batch_size=args.batch_size
        self.pc_valid=args.pc_valid
        self.root = args.data_folder
        # self.latent_dim = args.latent_dim
        self.use_memory = args.use_memory

        self.num_tasks = args.num_tasks
        self.num_classes = args.n_cls

        self.num_samples = args.mem_size

        self.inputsize = [3,64,64]
        mean = [0.4802, 0.4480, 0.3975]
        std = [0.2770, 0.2691, 0.2821]

        normalize = transforms.Normalize(mean=mean, std=std)

        train_transform = transforms.Compose([
            transforms.Resize(size=(args.size, args.size)),
            transforms.RandomResizedCrop(size=args.size, scale=(0.1 if args.dataset == 'tiny-imagenet' else 0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=args.size // 20 * 2 + 1, sigma=(0.1, 2.0))],
                                   p=0.5 if args.size > 32 else 0.0),
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
        # self.train_split = {}
        self.test_set = {}


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



    def get_stage2_train(self, task_id, opt):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        if task_id == 0:
            memory_classes = None
            memory=None
        else:
            memory_classes = self.task_ids
            memory = self.task_memory


        self.train_set[task_id] = iTinyImagenet(root=self.root, classes=self.task_ids[task_id],
                                                memory_classes=memory_classes, memory=memory,
                                                task_num=task_id, train=True, transform=self.transformation)

        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id],
                                                                 [len(self.train_set[task_id]) - split, split])
        if self.use_memory:
            self.train_set[task_id].update_memory()
            reply_indices = np.where(np.array(train_split.dataset.train_tt) < task_id)[0].tolist()
            train_split.indices = train_split.indices + reply_indices
        # self.train_split[task_id] = train_split




        weights = np.array([0.] * len(self.train_set[task_id].train_data))
        ut, uc = np.unique(self.train_set[task_id].train_global_labels, return_counts=True)

        # weight_list = np.array([0.]) * len(ut)
        for t, c in zip(ut, uc):
            weights[self.train_set[task_id].train_global_labels == t] = 1. / c


        train_sampler = data.WeightedRandomSampler(torch.Tensor(weights), len(weights))


        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory, sampler=train_sampler)


        self.dataloaders[task_id]['train'] = train_loader

        self.dataloaders[task_id]['name'] = 'TinyImageNet-{}-{}'.format(task_id,self.task_ids[task_id])


        print ("Task ID: ", task_id)
        print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        if self.use_memory and self.num_samples > 0 :
            self.update_memory(task_id)
        return self.dataloaders

    def get(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        if task_id == 0:
            memory_classes = None
            memory=None
        else:
            memory_classes = self.task_ids
            memory = self.task_memory


        self.train_set[task_id] = iTinyImagenet(root=self.root, classes=self.task_ids[task_id],
                                                memory_classes=memory_classes, memory=memory,
                                                task_num=task_id, train=True, transform=self.transformation)

        # self.test_set[task_id] = iMiniImageNet(root=self.root, classes=self.task_ids[task_id], memory_classes=None,
        #                                 memory=None, task_num=task_id, train=False, transform=self.transformation)




        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id], [len(self.train_set[task_id]) - split, split])
        if self.use_memory:
            train_split.dataset.update_memory()
            reply_indices = np.where(np.array(train_split.dataset.train_tt) < task_id)[0].tolist()
            train_split.indices = train_split.indices + reply_indices
        # self.train_split[task_id] = train_split

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=self.batch_size,
                                                   num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        # test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
        #                                           pin_memory=self.pin_memory, shuffle=True)


        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        # self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = 'MiniImageNet-{}-{}'.format(task_id,self.task_ids[task_id])
        # self.dataloaders[task_id]['tsne'] = torch.utils.data.DataLoader(self.test_set[task_id],
        #                                                                 batch_size=len(test_loader.dataset),
        #                                                                 num_workers=self.num_workers,
        #                                                                 pin_memory=self.pin_memory, shuffle=True)

        print ("Task ID: ", task_id)
        print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Validation set size: {} images of {}x{}".format(len(valid_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Train+Val  set size: {} images of {}x{}".format(len(valid_loader.dataset)+len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        # print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        if self.use_memory and self.num_samples > 0 and task_id != self.num_tasks-1:
            self.update_memory(task_id)


        return self.dataloaders

    def get_train(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        if task_id == 0:
            memory_classes = None
            memory=None
        else:
            memory_classes = self.task_ids
            memory = self.task_memory


        self.train_set[task_id] = iTinyImagenet(root=self.root, classes=self.task_ids[task_id],
                                                memory_classes=memory_classes, memory=memory,
                                                task_num=task_id, train=True, transform=self.transformation)

        # self.test_set[task_id] = iMiniImageNet(root=self.root, classes=self.task_ids[task_id], memory_classes=None,
        #                                 memory=None, task_num=task_id, train=False, transform=self.transformation)


        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id], [len(self.train_set[task_id]) - split, split])
        # self.train_split[task_id] = train_split

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=self.batch_size,
                                                   num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        # test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
        #                                           pin_memory=self.pin_memory, shuffle=True)

        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        # self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = 'iMiniImageNet-{}-{}'.format(task_id,self.task_ids[task_id])
        # self.dataloaders[task_id]['tsne'] = torch.utils.data.DataLoader(self.test_set[task_id],
        #                                                                 batch_size=len(test_loader.dataset),
        #                                                                 num_workers=self.num_workers,
        #                                                                 pin_memory=self.pin_memory, shuffle=True)

        print ("Task ID: ", task_id)
        print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Validation set size: {} images of {}x{}".format(len(valid_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Train+Val  set size: {} images of {}x{}".format(len(valid_loader.dataset)+len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        # print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        if self.use_memory and self.num_samples > 0 :
            self.update_memory(task_id)


        return self.dataloaders

    def get_test_dataloader(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        self.test_set[task_id] = iTinyImagenet(root=self.root, classes=self.task_ids[task_id], memory_classes=None,
                                           memory=None, task_num=task_id, train=False, transform=self.transformation)

        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                  pin_memory=self.pin_memory,shuffle=True)

        self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = 'TinyImageNet-{}-{}'.format(task_id,self.task_ids[task_id])
        print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        return self.dataloaders

    def get_all_test_dataloader(self):

        self.dataloaders = {}
        sys.stdout.flush()

        self.test_set = iTinyImagenet(root=self.root, classes=list(np.arange(100)),memory_classes=None, task_num=19,
                                           memory=None, train=False, transform=self.test_transformation)

        test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers,
                                                  pin_memory=self.pin_memory,shuffle=False)

        self.dataloaders = test_loader

        print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        return self.dataloaders

    def update_memory(self, task_id):
        num_classes_per_task = int(self.num_classes/self.num_tasks)
        # num_samples_per_class = self.num_samples // (int(self.num_classes/self.num_tasks) * (task_id+1))
        num_samples_per_class_previous = math.floor(self.num_samples / (num_classes_per_task * (task_id + 1)))
        num_samples_per_class_new = math.ceil((self.num_samples - (num_classes_per_task * task_id) * num_samples_per_class_previous) / num_classes_per_task)
        mem_class_mapping = {c: i for i, c in enumerate(self.task_ids[task_id])}

        # val_observed_targets = self.task_ids[task_id]
        # val_unique_cls = np.unique(val_observed_targets)  # 单个样本类别标签


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
                self.task_memory[i][self.task_ids[i][j]]['x'] = self.task_memory[i][self.task_ids[i][j]]['x'][:num_samples_per_class_previous]
                self.task_memory[i][self.task_ids[i][j]]['g_y'] = self.task_memory[i][self.task_ids[i][j]]['g_y'][:num_samples_per_class_previous]
                self.task_memory[i][self.task_ids[i][j]]['y'] = self.task_memory[i][self.task_ids[i][j]]['y'][:num_samples_per_class_previous]
                self.task_memory[i][self.task_ids[i][j]]['tt'] = self.task_memory[i][self.task_ids[i][j]]['tt'][:num_samples_per_class_previous]
                self.task_memory[i][self.task_ids[i][j]]['td'] = self.task_memory[i][self.task_ids[i][j]]['td'][:num_samples_per_class_previous]


        print ('Memory updated.')