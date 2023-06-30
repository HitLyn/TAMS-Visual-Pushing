import numpy as np
import os
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


def image_loader(path):
    return Image.open(path)

class VaeImageDataset(Dataset):
    def __init__(self, base_path, n_samples = 4e4, train = True, split = False, transform = None, loader = image_loader):
        """
            base_path: path/to/generated_images, which can get the full path: path/to/generated_images
            n_samples: how many samples are there in the training set
            train: bool, is this dataset for training or evaluation
            loader: function, how to load the images
        """
        self.n_samples = n_samples
        self.base_path = base_path
        self.transform = transform
        self.train = train
        self.transform = transform
        self.loader = image_loader
        self.split = split

    def __getitem__(self, index):
        if self.split:
            # get image starting idx
            if self.train:
                starting_idx = 0
            else:
                starting_idx = int(0.8*self.n_samples)
        else:
            starting_idx = 0

        file_name = "{:05d}.png".format(index + starting_idx)
        image = self.loader(os.path.join(self.base_path, file_name))
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        if self.split:
            if self.train:
                return int(0.8*self.n_samples)
            else:
                return int(0.2*self.n_samples)
        else:
            return int(self.n_samples)

