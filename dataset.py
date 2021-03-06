import os
import cv2
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset

class FaceDataset(Dataset):
    """
    dataset

    """

    def __init__(self, csv_path, transform=None):
        """
        initiate function
        :param csv_path: anotated data csvfile
        :param transform: itarational transformer
        """
        self.csv_path = csv_path
        self.transform = transform
        self.data = pd.read_csv(self.csv_path, header=None)
        self.image_num = len(self.data)
        self.label_dict = {}
        label_count = 0
        for i in range(self.image_num):
            if not self.data.ix[i, 1] in self.label_dict:
                self.label_dict[self.data.ix[i, 1]] = label_count
                label_count += 1

    def label_num(self):
        return len(self.label_dict)

    def get_label_dict(self):
        return self.label_dict

    def give_label_dict(self, label_dict):
        self.label_dict = label_dict

    def __len__(self):
        """
        return the data_num
        """
        return self.image_num

    def __getitem__(self, idx):
        image_path = self.data.ix[idx, 0]
        label = self.data.ix[idx, 1]
        label_onehot = np.zeros(len(self.label_dict))
        if not label in self.label_dict:
            label_onehot[-1] = 1
        else:
            label_onehot[self.label_dict[label]] = 1
        label_onehot = torch.tensor(label_onehot, dtype=torch.float)
        input_image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(input_image)
        return (image, label_onehot)

class FaceResolutorDataset(FaceDataset):

    def __getitem__(self, idx):
        higher_path = self.data.ix[idx, 0]
        lower_path = self.data.ix[idx, 1]
        higher_image = cv2.imread(higher_path)
        lower_image = cv2.imread(lower_path)
        if self.transform:
            higher_image = self.transform(higher_image)
            lower_image = self.transform(lower_image)
        return(lower_image, higher_image)

