import os
import cv2
import torch
import torchvision


def face_detection(image_file):
    cascade_path = "./models/haarcascade_frontalface_default.xml"

    image = cv2.imread(image_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BG2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)

    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

    color = (255, 255, 255)
    if len(facerect) > 0:

        for rect in facerect:
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)
            cv2.imshow(image)

class Dataset(torch.utils.data.Dataset):
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
