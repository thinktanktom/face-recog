import os
from cv2 import cv2
import detect_face
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    for image_name in dirs:
        image_path = data_folder_path + "/" + image_name
        image = cv2.imread(image_path)
        cv2.imshow("Training on image...",image)
        cv2.waitKey(100)
        face,rect = detect_face.detect_face(image)
        if face is not None:
            faces.append(face)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return faces