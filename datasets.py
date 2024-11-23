import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import sklearn
from ucimlrepo import fetch_ucirepo
import pandas as pd
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST


class Regression(Dataset):

    def __init__(self, ood, device):

        self.all_x = torch.from_numpy(np.loadtxt("data/snelson_data/train_inputs")).float().to(device).view(-1, 1)
        self.all_y = torch.from_numpy(np.loadtxt("data/snelson_data/train_outputs")).float().to(device).view(-1, 1)

        if ood:
            idx_test = torch.nonzero((self.all_x > 1.5) * (self.all_x < 3))[:,0]
            idx_train = torch.nonzero(~((self.all_x > 1.5) * (self.all_x < 3)))[:,0]
            self.x_test = self.all_x[idx_test]
            self.y_test = self.all_y[idx_test]
            self.x_train = self.all_x[idx_train]
            self.y_train = self.all_y[idx_train]
        else:
            idx = np.arange(self.all_x.shape[0])
            np.random.shuffle(idx)
            self.x_test = self.all_x[idx[150:]]
            self.y_test = self.all_y[idx[150:]]
            self.x_train = self.all_x[idx[:150]]
            self.y_train = self.all_y[idx[:150]]

        idx_sort = torch.argsort(self.x_test[:,0])
        self.x_test = self.x_test[idx_sort]
        self.y_test = self.y_test[idx_sort]

        idx_sort = torch.argsort(self.all_x[:,0])
        self.all_x = self.all_x[idx_sort]
        self.all_y = self.all_y[idx_sort]

        self.x_test_unlabeled = torch.from_numpy(np.loadtxt("data/snelson_data/test_inputs")).float().to(device).view(-1, 1)

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def get_test(self):
        return self.x_test, self.y_test


class Banana(Dataset):

    def __init__(self, device):

        Xy = np.loadtxt("data/banana/banana.csv", delimiter=",")
        x_full = torch.from_numpy(Xy[:, :-1]).float().to(device)
        y_full = torch.from_numpy(Xy[:, -1]).long().to(device) - 1

        split_train_size = 0.7
        N = x_full.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        self.x_train = x_full[idx[:int(split_train_size*N)]]
        self.y_train = y_full[idx[:int(split_train_size * N)]]
        self.x_test = x_full[idx[int(split_train_size * N):]]
        self.y_test = y_full[idx[int(split_train_size * N):]]

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def get_test(self):
        return self.x_test, self.y_test


class Uci(Dataset):

    def __init__(self, dataset_type, device):

        if dataset_type == 'UCI_australian':
            statlog_australian_credit_approval = fetch_ucirepo(id=143)
            X = statlog_australian_credit_approval.data.features
            y = statlog_australian_credit_approval.data.targets
        elif dataset_type == 'UCI_breast':
            breast_cancer = fetch_ucirepo(id=14)
            X = breast_cancer.data.features
            y = breast_cancer.data.targets
        elif dataset_type == 'UCI_glass':
            glass_identification = fetch_ucirepo(id=42)
            X = glass_identification.data.features
            y = glass_identification.data.targets
        elif dataset_type == 'UCI_ionosphere':
            ionosphere = fetch_ucirepo(id=52)
            X = ionosphere.data.features
            y = ionosphere.data.targets
        elif dataset_type == 'UCI_vehicle':
            in_vehicle_coupon_recommendation = fetch_ucirepo(id=603)
            X = in_vehicle_coupon_recommendation.data.features
            y = in_vehicle_coupon_recommendation.data.targets
        elif dataset_type == 'UCI_waveform':
            waveform_database_generator_version_1 = fetch_ucirepo(id=107)
            X = waveform_database_generator_version_1.data.features
            y = waveform_database_generator_version_1.data.targets
        else:
            X, y = None, None
            print("Error, wrong dataset type specified")
            exit()

        X = X.copy()
        for col in X.select_dtypes(include='object').columns:
            X[col] = pd.Categorical(X[col])
        for col in X.select_dtypes(include='category').columns:
            X[col] = X[col].cat.codes
        y = y.copy()
        for col in y.select_dtypes(include='object').columns:
            y[col] = pd.Categorical(y[col])
        for col in y.select_dtypes(include='category').columns:
            y[col] = y[col].cat.codes

        X = torch.from_numpy(X.values).float().to(device)
        y = torch.from_numpy(y.values).long().to(device)

        split_train_size = 0.7
        N = X.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        self.x_train = X[idx[:int(split_train_size*N)]]
        self.y_train = y[idx[:int(split_train_size * N)]]
        self.x_test = X[idx[int(split_train_size * N):]]
        self.y_test = y[idx[int(split_train_size * N):]]

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def get_test(self):
        return self.x_test, self.y_test


class Mnist(Dataset):

    def __init__(self, dataset_type, device):

        self.device = device

        data = MNIST if dataset_type == 'MNIST' else FashionMNIST

        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = data(root='data', train=True, transform=transform, download=True)
        valid_test_dataset = data(root='data', train=False, transform=transform, download=True)

        x_train = train_dataset.train_data
        y_train = train_dataset.train_labels
        x_test = valid_test_dataset.train_data
        y_test = valid_test_dataset.train_labels

        # subsample dataset uniformely on the labels distribution to approximately 5000 data points
        x_train_sub, y_train_sub = [], []
        for i in range(10):
            idx_i = torch.nonzero(y_train == i).numpy()
            idx_i = idx_i[:int(0.0835*idx_i.shape[0]), 0]
            x_train_sub.append(x_train[idx_i])
            y_train_sub.append(y_train[idx_i])
        x_train_sub = torch.cat(x_train_sub, 0)
        y_train_sub = torch.cat(y_train_sub, 0)

        self.x_train = torch.unsqueeze(x_train_sub.float(), 1).to(self.device)
        self.y_train = y_train_sub.long().to(self.device)
        self.x_test = torch.unsqueeze(x_test.float(), 1).to(self.device)
        self.y_test = y_test.long().to(self.device)

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def get_test(self):
        return self.x_test, self.y_test


