import torch
from torch.utils.data import TensorDataset, DataLoader


class Loader:
    def __init__(self, X, Y, batch_size, shuffle=False, drop_last=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        # Convert the datapoints into tensors
        features = torch.Tensor(X)
        targets = torch.Tensor(Y)

        # Combine them using TensorDataset class
        self.dataset = TensorDataset(features, targets)

    def get_loader(self):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return data_loader

    def get_one_loader(self):
        one_data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=True)
        return one_data_loader


def get_train_loader(X_train, Y_train, batch_size, shuffle=False, drop_last=True):
    train_load = Loader(X_train, Y_train, batch_size, shuffle, drop_last)
    return train_load.get_loader()


def get_val_loader(X_val, Y_val, batch_size, shuffle=False, drop_last=True):
    val_load = Loader(X_val, Y_val, batch_size, shuffle, drop_last)
    return val_load.get_loader()
