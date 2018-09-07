import os
import re
import cv2
import pandas as pd
import numpy as np


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
    image_shape = (256, 256)
    image_list = os.listdir(image_dir)
    count_image = 0
    for image_file in image_list:
        image_path = image_dir + "/" + image_file
        print(image_path)
        image = imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_path)

        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        if len(facerect) == 1:

            for rect in facerect:
                resize_image = cv2.resize(image[rect[1]:rect[1]+rect[3], rect[0]:rect[0] + rect[2]], image_shape)
                cv2.imwrite(output_dir + "/saito_" + str(count_image) + ".jpg", resize_image)
                count_image += 1

def dir_rename(image_dir, output_dir):
    image_list = os.listdir(image_dir)
    for i, image_file in enumerate(image_list):
        os.rename(image_dir + "/" + image_file, output_dir + "/" + re.sub("[0-9]+", str(i), image_file))

def make_dataset_csv(image_dir, output_csv):
    image_list = os.listdir(image_dir)
    csv_list = []
    for image_file in image_list:
        csv_list.append(( image_dir + "/" + image_file, image_file.split("_")[0]))
    df = pd.DataFrame(csv_list)
    df.to_csv(output_csv, index=False, header=False)


if __name__ == "__main__":
    image_dir = "./saito_asuka"
    output_dir = "./saito_face"
    #face_detection(image_dir, output_dir)
    #dir_rename(output_dir, "tmp")
    #dir_rename("tmp", output_dir )
    make_dataset_csv("./shiraishi_saito", "./data.csv")

