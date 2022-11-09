import os

import numpy as np
import pandas as pd
import torchvision
import torch


def load_imagenet50():
    x_train = torch.load('./datasets/MOCO/IMAGENET_50/train_codes.pt')
    x_test = torch.load('./datasets/MOCO/IMAGENET_50/val_codes.pt')
    y_train = torch.load('./datasets/MOCO/IMAGENET_50/train_labels.pt')
    y_test = torch.load('./datasets/MOCO/IMAGENET_50/val_labels.pt')
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    return x, y


def load_stl10():
    x_train = torch.load('./datasets/MOCO/STL10/train_codes.pt')
    x_test = torch.load('./datasets/MOCO/STL10/val_codes.pt')
    y_train = torch.load('./datasets/MOCO/STL10/train_labels.pt')
    y_test = torch.load('./datasets/MOCO/STL10/val_labels.pt')
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    return x, y



def load_emnist_by():
    x_train = torchvision.datasets.EMNIST(root='./datasets', train=True, download=True, split='byclass')
    x_test =  torchvision.datasets.EMNIST(root='./datasets', train=False, download=True, split='byclass')
    x = np.concatenate((x_train.data, x_test.data))
    y = np.concatenate((x_train.targets, x_test.targets))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    return x, y


def load_cifar_10():
    x_train = torch.load('./datasets/simclr/cifar10/train_codes.pt').cpu().numpy()
    x_test = torch.load('./datasets/simclr/cifar10/val_codes.pt').cpu().numpy()
    y_train = torch.load('./datasets/simclr/cifar10/train_labels.pt').cpu().numpy()
    y_test = torch.load('./datasets/simclr/cifar10/val_labels.pt').cpu().numpy()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    return x, y

def load_cifar_100():
    x_train = torch.load('./datasets/simclr/cifar100/train_codes.pt').cpu().numpy()
    x_test = torch.load('./datasets/simclr/cifar100/val_codes.pt').cpu().numpy()
    y_train = torch.load('./datasets/simclr/cifar100/train_labels.pt').cpu().numpy()
    y_test = torch.load('./datasets/simclr/cifar100/val_labels.pt').cpu().numpy()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    return x, y



def load_mnist():
    x_train = torchvision.datasets.MNIST(root='./datasets', train=True, download=True)
    x_test =  torchvision.datasets.MNIST(root='./datasets', train=False, download=True)
    x = np.concatenate((x_train.data, x_test.data))
    y = np.concatenate((x_train.targets, x_test.targets))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    return x, y


def load_mnist_test():
    x_test =  torchvision.datasets.MNIST(root='./datasets', train=False, download=True)
    x = x_test.data
    y = x_test.targets
    x = np.divide(x, 255.)
    x = x.reshape((x.shape[0], -1))
    return x, y


def load_fashion():
    x_train = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True)
    x_test =  torchvision.datasets.FashionMNIST(root='./datasets', train=False, download=True)
    x = np.concatenate((x_train.data, x_test.data))
    y = np.concatenate((x_train.targets, x_test.targets))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    y_names = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
               5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
    return x, y, y_names


def load_har():
    x_train = pd.read_csv(
        'data/har/train/X_train.txt',
        sep=r'\s+',
        header=None)
    y_train = pd.read_csv('data/har/train/y_train.txt', header=None)
    x_test = pd.read_csv('data/har/test/X_test.txt', sep=r'\s+', header=None)
    y_test = pd.read_csv('data/har/test/y_test.txt', header=None)
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    # labels start at 1 so..
    y = y - 1
    y = y.reshape((y.size,))
    y_names = {0: 'Walking', 1: 'Upstairs', 2: 'Downstairs', 3: 'Sitting', 4: 'Standing', 5: 'Laying', }
    return x, y, y_names


def load_usps(data_path='data/usps'):
    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64')
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y


def load_pendigits(data_path='data/pendigits'):
    if not os.path.exists(data_path + '/pendigits.tra'):
        os.makedirs(data_path,  exist_ok=True)
        
        os.system(
            'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra -P %s' %
            data_path)
        os.system(
            'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes -P %s' %
            data_path)
        os.system(
                'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.names -P %s' %
            data_path)

    # load training data
    with open(data_path + '/pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]

    # load testing data
    with open(data_path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    y = y.astype('int')
    return x, y

def load_reuters(data_path='data/REUTERS'):
    x = np.load(data_path + '/10k_feature.npy').astype(np.float32)
    y = np.load(data_path + '/10k_target.npy').astype(np.int32)
    return x, y

