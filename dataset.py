import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def face_detection(image_dir, output_dir):
    cascade_path = "./haarcascade_frontalface_default.xml"
    color = (255, 255, 255)
    image_shape = (255, 255)
    image_list = os.listdir(image_dir)
    count_image = 0
    for image_file in image_list:
        image_path = image_dir + "/" + image_file
        image = imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_path)

        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(facerect) > 0:

            for rect in facerect:
                cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)
                resize_image = cv2.resize(image[rect[1]:rect[1]+rect[3], rect[0]:rect[0] + rect[2]], image_shape)
                cv2.imwrite(output_dir + "/白石麻衣_" + str(count_image) + ".jpg", resize_image)


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


if __name__ == "__main__":
    image_dir = "./shiraishi_mai"
    output_dir = "./shiraishi_face"
    face_detection(image_dir, output_dir)
