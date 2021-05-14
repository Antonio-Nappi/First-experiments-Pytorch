import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self):
        xy = np.loadtxt("./data/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, 0])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


# dataset can be also transformed, so let's implement this feature to our dataset
class WineDatasetTransform(Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt("./data/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, 0]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        return self.factor * sample[0], sample[1]


if __name__ == "__main__":
    transform = False
    # print to check if everything is ok
    if not transform:
        dataset = WineDataset()
        first_data = dataset[0]
        features, labels = first_data
        print(features, labels)
        # create dataloader
        batch_size = 4
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        dataiter = iter(dataloader)
        data = dataiter.next()
        features, labels = data
        print(features, labels)

        num_epochs = 2
        total_samples = len(dataset)
        n_iter = math.ceil(total_samples / batch_size)
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(dataloader):
                if (i + 1) % 5 == 0:
                    print(f'epoch:{epoch + 1}/{num_epochs}, step{i + 1}/{n_iter}, inputs {inputs.shape}')
    else:
        dataset = WineDatasetTransform(transform=ToTensor())
        first_data = dataset[0]
        features, labels = first_data
        print(features, labels)
        composed = torchvision.transforms.Compose(ToTensor(), MulTransform(2))
        dataset = WineDatasetTransform(transform=composed)
        first_data = dataset[0]
        features, labels = first_data
        print(features, labels)
