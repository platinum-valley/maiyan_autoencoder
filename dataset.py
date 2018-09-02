import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class face_train_Dataset(Dataset):
    """
    データセット

    """

    def __init__(self, image_dir, image_name="shiraishi", transform=None):
        """
        初期化関数
        :param image_dir: 画像のルートディレクトリ
        :param image_name: 画像の名前 "[image_name](番号).jpg"
        """
        self.image_dir = image_dir
        self.image_name = image_name
        self.image_num = len(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        """
        データセットの総数を返す
        :return:
        """
        return self.image_num

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_name + "_" + str(idx) + ".jpg")
        input_image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(input_image)
        return (image, image)
