from cv2 import cv2
def load_img(path):
    image = cv2.imread(path)
    
    return image[...,::-1]