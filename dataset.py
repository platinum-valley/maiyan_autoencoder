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


def face_detection(image_file):
    if not os.path.exists(image_file):
        print("not found image file")
        return None
    cascade_path = "./haarcascade_frontalface_default.xml"

    image = imread(image_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)

    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

    color = (255, 255, 255)
    if len(facerect) > 0:

        for rect in facerect:
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)
        #cv2.imshow("test", image[rect[1]:rect[1]+rect[3], rect[0]:rect[0] + rect[2]])
        cv2.imshow("test", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
    test = face_detection("./shiraishi_mai/白石麻衣_137.jpg")
