import os
import cv2
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset

class face_train_Dataset(Dataset):
    """
    データセット

    """

    def __init__(self, image_dir, csv_path, transform=None):
        """
        初期化関数
        :param image_dir: 画像のルートディレクトリ
        :param csv_path: 入力と出力が対応したcsvファイルのパス
        :param transform: トランスフォーマ
        """
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.image_num = len(os.listdir(image_dir))
        self.transform = transform
        self.data = pd.read_csv(self.csv_path, header=None)

    def __len__(self):
        """
        データセットの総数を返す
        :return:
        """
        return self.image_num

    def __getitem__(self, idx):
        image_path = self.data.ix[idx, 0]
        print(image_path)
        input_image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(input_image)
        return (image, image)
