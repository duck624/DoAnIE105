import os
import torch
from PIL import Image
from torchvision.datasets import MNIST as TorchMNIST, CIFAR100 as TorchCIFAR100
import numpy as np

class MNIST:
    def __init__(self, root, indices=None, train=True, download=False, transform=None, target_transform=None, need_index=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.need_index = need_index
        self.indices = indices

        # Load torchvision MNIST dataset
        self.mnist_dataset = TorchMNIST(root=self.root, train=self.train, download=download)

        if self.indices is not None:
            self.data = self.mnist_dataset.data[indices]
            self.targets = self.mnist_dataset.targets[indices]
            self.true_index = np.array(indices)
        else:
            self.data = self.mnist_dataset.data
            self.targets = self.mnist_dataset.targets
            self.true_index = np.arange(len(self.data))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        i = self.true_index[index]
        img = Image.fromarray(img.numpy(), mode='L')  # Grayscale image
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.need_index:
            return img, target, i
        else:
            return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        # Leverage torchvision's MNIST check
        return self.mnist_dataset._check_exists()

    def download(self):
        # Download train and test sets
        self.mnist_dataset = TorchMNIST(root=self.root, train=True, download=True)
        self.mnist_test_dataset = TorchMNIST(root=self.root, train=False, download=True)

class CIFAR100:
    def __init__(self, root, indices=None, train=True, download=False, transform=None, target_transform=None, need_index=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.need_index = need_index
        self.indices = indices

        # Load torchvision CIFAR100 dataset
        self.cifar_dataset = TorchCIFAR100(root=self.root, train=self.train, download=download)

        if self.indices is not None:
            self.data = np.array(self.cifar_dataset.data)[indices]
            self.targets = np.array(self.cifar_dataset.targets)[indices]
            self.true_index = np.array(indices)
        else:
            self.data = self.cifar_dataset.data
            self.targets = self.cifar_dataset.targets
            self.true_index = np.arange(len(self.data))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        i = self.true_index[index]
        img = Image.fromarray(img)  # RGB image
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.need_index:
            return img, target, i
        else:
            return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        return self.cifar_dataset._check_exists()

    def download(self):
        self.cifar_dataset = TorchCIFAR100(root=self.root, train=True, download=True)
        self.cifar_test_dataset = TorchCIFAR100(root=self.root, train=False, download=True)
