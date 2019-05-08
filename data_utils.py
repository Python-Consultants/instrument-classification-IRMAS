import h5py
import numpy as np


def read_file(dataset):

    data, label = [], []

    h5f1 = h5py.File('data.h5', 'r')
    data = h5f1['dataset_1'][:]
    h5f2 = h5py.File('label.h5', 'r')
    label = h5f2['dataset_1'][:]

    np.random.seed(123)
    idx = np.arange(len(label))
    np.random.shuffle(idx)
    data = data[idx, :, :]
    label = label[idx]
    label = np.eye(11)[label]

    split_point = int(0.8 * len(label))
    train_data = data[:split_point]
    train_label = label[:split_point]
    valid_data = data[split_point:]
    valid_label = label[split_point:]
    if dataset == 'train':
        return train_data, train_label

    elif dataset == 'valid':
        return valid_data, valid_label

    elif dataset == 'test':
        return valid_data, valid_label
