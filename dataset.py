import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class face_Dataset(Dataset):
    """
    データセット

    """

    def __init__(self, image_dir, transform=None):
        """
        初期化関数
        :param image_dir: 画像のルートディレクトリ
        """
        self.image_dir = image_dir
        self.image_num = len(os.listdir(image_dir))
        pass

    def __len__(self):
        """
        データセットの総数を返す
        :return:
        """
        return len(Dataset)

    def __getitem__(self):
        pass
