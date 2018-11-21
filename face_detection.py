import os
import shutil
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


def face_detection(image_dir, output_dir, name):
    cascade_path = "./haarcascade_frontalface_default.xml"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
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

        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=30, minSize=(30, 30))

        if len(facerect) == 1:

            for rect in facerect:
                resize_image = cv2.resize(image[rect[1]:rect[1]+rect[3], rect[0]:rect[0] + rect[2]], image_shape)
                cv2.imwrite(output_dir + "/" + name + "_" + str(count_image) + ".jpg", resize_image)
                count_image += 1

def dir_rename(image_dir, output_dir):
    image_list = os.listdir(image_dir)
    for i, image_file in enumerate(image_list):
        os.rename(image_dir + "/" + image_file, output_dir + "/" + image_file.split("_")[0] + "_" + str(i) + ".jpg")

def make_dataset_csv(image_dir_list, output_csv):
    csv_list = []
    for image_dir in image_dir_list:
        image_list = os.listdir(image_dir)

        for image_file in image_list:
            csv_list.append((image_dir + "/" + image_file, image_file.split("_")[0]))
    df = pd.DataFrame(csv_list)
    df.to_csv(output_csv, index=False, header=False)

def flip_image(image):
    return image[:, ::-1, :]

def add_noise(image):
    noise = np.random.normal(0, 0.1, (256, 256, 3)) * 255
    return image + noise

def make_lower(image):
    image = cv2.resize(image, (32, 32))
    image = cv2.resize(image, (256, 256))
    return image


def augmentation(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for img in os.listdir(input_dir):
        input_img = cv2.imread(input_dir + "/" + img)
        fliped = flip_image(input_img)
        noisy = add_noise(input_img)
        cv2.imwrite(output_dir + "/" + img, input_img)
        cv2.imwrite(output_dir + "/" + img.split("_")[0] + "_flip_" + img.split("_")[1], fliped)
        cv2.imwrite(output_dir + "/" + img.split("_")[0] + "_noisy_" + img.split("_")[1], noisy)

if __name__ == "__main__":
    input_dirs = ["shiraishi_mai", "saito_asuka", "akimoto_manatsu", "hori_miona", "hoshino_minami", "ikoma_rina", "yamashita_mizuki", "yoda_yuki","ikuta_erika", "matsumura_sayuri", "nishino_nanase"]
    output_dirs = ["shiraishi_face", "saito_face", "akimoto_face", "hori_face", "hoshino_face", "ikoma_face", "yamashita_face", "yoda_face","ikuta_face", "matsumura_face", "nishino_face"]
    dataset_dir = "nogizaka_face"
    #for input, output in zip(input_dirs, output_dirs):
    #    face_detection("./"+input, "./"+output, "celeb")
    """
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    os.mkdir("tmp")
    for data_dir in output_dirs:
        augmentation(data_dir, data_dir + "_aug")
        dir_rename(data_dir + "_aug", "tmp")
        dir_rename("tmp", dataset_dir)
    shutil.rmtree("tmp")
    """
    #make_dataset_csv(["nogizaka_face"], "./train_data.csv")
    make_lower(cv2.imread("nogi_face/nishino_1.jpg"))
