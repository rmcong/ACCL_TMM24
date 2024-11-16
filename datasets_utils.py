# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import copy
import os.path

import sys
import warnings
import urllib.request

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import numpy as np

import torch
from torchvision import datasets, transforms

from utils import *
# from scipy.imageio import imread

import os
import torch
from PIL import Image
import scipy.io as sio
from collections import defaultdict
from itertools import chain
from collections import OrderedDict





class CIFAR10_(datasets.CIFAR10):

    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    num_classes = 10

    def __init__(self, root, task_num, num_samples_per_class, train, transform, target_transform, download=True, memory=None, memory_class=None):
        # root, task_num, train, transform = None, download = False):
        super(CIFAR10_, self).__init__(root, task_num, transform=transform,
                                        target_transform=target_transform,
                                        download=download)
        # print(self.train)
        # self.train = train  # training set or test set

        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform=target_transform
        self.task_id = task_num

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        if not num_samples_per_class:
            self.data = []
            self.targets = []

            # now load the picked numpy arrays
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, 'rb') as f:
                    if sys.version_info[0] == 2:
                        entry = pickle.load(f)
                    else:
                        entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    if 'labels' in entry:
                        self.targets.extend(entry['labels'])
                    else:
                        self.targets.extend(entry['fine_labels'])

        else:
            x, g_y, y, tt, td = [], [], [], [], []
            for l in range(self.num_classes):
                indices_with_label_l = np.where(np.array(self.targets)==l)

                x_with_label_l = [self.data[item] for item in indices_with_label_l[0]]
                y_with_label_l = [l]*len(x_with_label_l)

                # If we need a subset of the dataset with num_samples_per_class we use this and then concatenate it with a complete dataset
                shuffled_indices = np.random.permutation(len(x_with_label_l))[:num_samples_per_class]
                x_with_label_l = [x_with_label_l[item] for item in shuffled_indices]
                y_with_label_l = [y_with_label_l[item] for item in shuffled_indices]

                x.append(x_with_label_l)
                y.append(y_with_label_l)
                g_y.append(y_with_label_l+self.task_id*self.num_classes)

            self.data = np.array(sum(x,[]))
            self.targets = sum(y,[])
            self.train_global_labels = sum(g_y, [])


        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        self.memory_classes = memory_class
        self.memory = memory
        x, y, tt, td, g_y = [], [], [], [], []
        for i in range(len(self.data)):
            x.append(self.data[i])
            y.append(self.targets[i])
            tt.append(task_num)
            td.append(task_num+1)
            g_y.append(self.targets[i] + self.task_id * self.num_classes)

        self.train_global_labels = g_y
        self.train_data = x
        self.train_labels = y
        self.train_tt = tt
        self.train_td = td

    def update_memory(self):
        if self.memory_classes:
            for task_id in range(self.task_id):
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

        img, g_target, target, tt, td = self.train_data[index], self.train_global_labels[index], self.train_labels[index], self.train_tt[index], self.train_td[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img)
        except:
            pass

        try:
            if self.transform is not None: img = self.transform(img)
        except:
            pass
        try:
            if self.target_transform is not None: tt = self.target_transform(tt)
            if self.target_transform is not None: td = self.target_transform(td)
        except:
            pass

        return img, g_target, target, tt, td



    def __len__(self):
        # if self.train:
        return len(self.data)
        # else:
        #     return len(self.test_data)


    def report_size(self):
        print("CIFAR10 size at train={} time: {} ".format(self.train,self.__len__()))

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


class CIFAR100_(CIFAR10_):
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
    num_classes = 100


class SVHN_(torch.utils.data.Dataset):
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}


    def __init__(self, root, task_num, num_samples_per_class, train,transform=None, target_transform=None, download=True, memory=None, memory_class=None):
        self.root = os.path.expanduser(root)
        # root, task_num, train, transform = None, download = False):
        # print(self.train)
        # self.train = train  # training set or test set

        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform=target_transform
        self.task_id = task_num
        if self.train:
            split="train"
        else:
            split="test"

        self.num_classes = 10
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))


        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat['y'].astype(np.int64).squeeze() - 1

        self.data = np.transpose(self.data, (3, 2, 0, 1))

        if num_samples_per_class:
            x, g_y, y, tt, td = [], [], [], [], []
            for l in range(self.num_classes+1):
                indices_with_label_l = np.where(np.array(self.targets)==l)

                x_with_label_l = [self.data[item] for item in indices_with_label_l[0]]
                y_with_label_l = [l]*len(x_with_label_l)

                # If we need a subset of the dataset with num_samples_per_class we use this and then concatenate it with a complete dataset
                shuffled_indices = np.random.permutation(len(x_with_label_l))[:num_samples_per_class]
                x_with_label_l = [x_with_label_l[item] for item in shuffled_indices]
                y_with_label_l = [y_with_label_l[item] for item in shuffled_indices]

                x.append(x_with_label_l)
                y.append(y_with_label_l-1)
                g_y.append(y_with_label_l+self.task_id*self.num_classes)


            self.data = np.array(sum(x,[]))
            self.targets = np.array(sum(y,[])).astype(np.int64)
            self.train_global_labels = np.array(sum(g_y,[])).astype(np.int64)


        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        # np.place(self.targets, self.targets == 10, 0)

        # print ("svhn: ", self.data.shape)

        self.memory_classes = memory_class
        self.memory = memory
        x, y, tt, td, g_y = [], [], [], [], []
        for i in range(len(self.data)):
            x.append(self.data[i])
            y.append(self.targets[i])
            tt.append(task_num)
            td.append(task_num+1)
            g_y.append(self.targets[i] + self.task_id * self.num_classes)

        self.train_global_labels = g_y
        self.train_data = x
        self.train_labels = y
        self.train_tt = tt
        self.train_td = td

    def update_memory(self):
        if self.memory_classes:
            for task_id in range(self.task_id):
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
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, g_target, target, tt, td = self.train_data[index], int(self.train_global_labels[index]), int(self.train_labels[index]), self.train_tt[index], self.train_td[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        except:
            pass

        try:
            if self.transform is not None: img = self.transform(img)
        except:
            pass
        try:
            if self.target_transform is not None: tt = self.target_transform(tt)
            if self.target_transform is not None: td = self.target_transform(td)
        except:
            pass

        return img, g_target, target, tt, td


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


class MNIST_RGB(datasets.MNIST):

    def __init__(self, root, task_num, num_samples_per_class, train=True, transform=None, target_transform=None, download=False, memory=None, memory_class=None):
        super(MNIST_RGB, self).__init__(root, task_num, transform=transform,
                                        target_transform=target_transform,
                                        download=download)
        self.train = train  # training set or test set
        self.target_transform=target_transform
        self.transform=transform
        self.num_classes=10
        self.task_id = task_num
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        # self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.data=np.array(self.data).astype(np.float32)
        self.targets=list(np.array(self.targets))

        if num_samples_per_class:
            x, y, tt, td, g_y = [], [], [], [], []
            for l in range(self.num_classes):
                indices_with_label_l = np.where(np.array(self.targets)==l)

                x_with_label_l = [self.data[item] for item in indices_with_label_l[0]]
                # y_with_label_l = [l]*len(x_with_label_l)

                # If we need a subset of the dataset with num_samples_per_class we use this and then concatenate it with a complete dataset
                shuffled_indices = np.random.permutation(len(x_with_label_l))[:num_samples_per_class]
                x_with_label_l = [x_with_label_l[item] for item in shuffled_indices]
                y_with_label_l = [l]*len(shuffled_indices)

                x.append(x_with_label_l)
                y.append(y_with_label_l)
                g_y.append(y_with_label_l+self.task_id*self.num_classes)

            self.data = np.array(sum(x,[]))
            self.targets = sum(y,[])
            self.train_global_labels = sum(g_y,[])

        self.memory_classes = memory_class
        self.memory = memory

        x, y, tt, td, g_y = [], [], [], [], []
        for i in range(len(self.data)):
            x.append(self.data[i])
            y.append(self.targets[i])
            tt.append(task_num)
            td.append(task_num+1)
            g_y.append(self.targets[i] + self.task_id * self.num_classes)

        self.train_global_labels = g_y
        self.train_x = x
        self.train_y = y
        self.train_tt = tt
        self.train_td = td

    def update_memory(self):
        if self.memory_classes:
            for task_id in range(self.task_id):
                for j in self.memory_classes[task_id]:
                    len_memory = len(self.memory[task_id][j]['x'])
                    if len_memory == 0:
                        pass
                    else:
                        for k in range(len_memory):
                            self.train_x.append(self.memory[task_id][j]['x'][k])
                            self.train_global_labels.append(self.memory[task_id][j]['g_y'][k])
                            self.train_y.append(self.memory[task_id][j]['y'][k])
                            self.train_tt.append(self.memory[task_id][j]['tt'][k])
                            self.train_td.append(self.memory[task_id][j]['td'][k])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, g_target, target, tt, td = self.train_x[index], int(self.train_global_labels[index]), int(self.train_y[index]), self.train_tt[index], self.train_td[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img).convert('RGB')
        except:
            pass

        try:
            if self.transform is not None: img = self.transform(img)
        except:
            pass
        try:
            if self.target_transform is not None: tt = self.target_transform(tt)
            if self.target_transform is not None: td = self.target_transform(td)
        except:
            pass

        return img, g_target, target, tt, td



    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class FashionMNIST_(MNIST_RGB):
    """`Fashion MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]


class notMNIST_(torch.utils.data.Dataset):

    def __init__(self, root, task_num, num_samples_per_class, train,transform=None, target_transform=None, download=False, memory=None, memory_class=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train
        self.task_id = task_num

        self.url = "https://github.com/facebookresearch/Adversarial-Continual-Learning/blob/main/data/notMNIST.zip"
        self.filename = 'notMNIST.zip'
        # fpath = os.path.join(root, self.filename)
        # if not os.path.isfile(fpath):
        #     if not download:
        #        raise RuntimeError('Dataset not found. You can use download=True to download it')
        #     else:
        #         print('Downloading from '+self.url)
        #         download_url(self.url, root, filename=self.filename)

        # import zipfile
        # zip_ref = zipfile.ZipFile(fpath, 'r')
        # zip_ref.extractall(root)
        # zip_ref.close()

        if self.train:
            fpath = os.path.join(root, 'notMNIST', 'Train')

        else:
            fpath = os.path.join(root, 'notMNIST', 'Test')


        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

        self.num_classes = len(set(self.targets))


        if num_samples_per_class:
            x, g_y, y, tt, td = [], [], [], [], []
            for l in range(self.num_classes):
                indices_with_label_l = np.where(np.array(self.targets)==l)

                x_with_label_l = [self.data[item] for item in indices_with_label_l[0]]

                # If we need a subset of the dataset with num_samples_per_class we use this and then concatenate it with a complete dataset
                shuffled_indices = np.random.permutation(len(x_with_label_l))[:num_samples_per_class]
                x_with_label_l = [x_with_label_l[item] for item in shuffled_indices]
                y_with_label_l = [l]*len(shuffled_indices)

                x.append(x_with_label_l)
                y.append(y_with_label_l)
                g_y.append(y_with_label_l+self.task_id*self.num_classes)

            self.data = np.array(sum(x,[]))
            self.targets = sum(y,[])
            self.train_global_labels = sum(y, [])

        self.memory_classes = memory_class
        self.memory = memory
        x, y, tt, td, g_y = [], [], [], [], []
        for i in range(len(self.data)):
            x.append(self.data[i])
            y.append(self.targets[i])
            tt.append(task_num)
            td.append(task_num+1)
            g_y.append(self.targets[i] + self.task_id * self.num_classes)

        self.train_global_labels = g_y
        self.train_data = x
        self.train_labels = y
        self.train_tt = tt
        self.train_td = td

    def update_memory(self):
        if self.memory_classes:
            for task_id in range(self.task_id):
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
        img, g_target, target, tt, td = self.train_data[index], self.train_global_labels[index], self.train_labels[index], self.train_tt[index], self.train_td[index]

        try:
            img = Image.fromarray(img)
        except:
            pass

        try:
            if self.transform is not None:
                img = self.transform(img)
        except:
            pass
        # img = Image.fromarray(img)#.convert('RGB')
        # img = self.transform(img)

        return img, g_target, target, tt, td

    def __len__(self):
        return len(self.data)

    def download(self):
        """Download the notMNIST data if it doesn't exist in processed_folder already."""

        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()






