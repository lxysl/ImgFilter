import torch
from torch.utils.data import Dataset
from PIL import Image


class pairDataset(Dataset):
    def __init__(self, path_pair, label, transform):
        self.data = path_pair  # str
        self.label = label  # array
        self.transform = transform

    def __getitem__(self, index):
        # load jpg image from path
        img1 = Image.open(self.data[index][0]).convert('RGB')
        img2 = Image.open(self.data[index][1]).convert('RGB')

        # transform image
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = self.label[index]  # int
        return img1, img2, label

    def __len__(self):
        return len(self.data)
