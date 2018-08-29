import os
import cv2
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
        image = imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_path)

        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=30, minSize=(30, 30))

        if len(facerect) == 1:

            for rect in facerect:
                resize_image = cv2.resize(image[rect[1]:rect[1]+rect[3], rect[0]:rect[0] + rect[2]], image_shape)
                cv2.imwrite(output_dir + "/shiraishi_" + str(count_image) + ".jpg", resize_image)
                count_image += 1


if __name__ == "__main__":
    image_dir = "./shiraishi_mai"
    output_dir = "./shiraishi_face"
    face_detection(image_dir, output_dir)
