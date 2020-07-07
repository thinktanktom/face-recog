from cv2 import cv2
import os
import load_img
import detect_face
import prepare_training_data
import draw_rectangle
import predict
real = "/home/lucid/work/facial_recog/data/real_and_fake_face/training_real/"

real_path = os.listdir(real)
print("Preparing data...")
faces = prepare_training_data.prepare_training_data("/home/lucid/work/facial_recog/data/real_and_fake_face/training_real/")
print("Data prepared")
print("Predicting images...")
test_img1 = cv2.imread("/home/lucid/work/facial_recog/data/real_and_fake_face/training_real/real_00001.jpg")
test_img2 = cv2.imread("/home/lucid/work/facial_recog/data/real_and_fake_face/training_real/real_00002.jpg")
predicted_img1 = predict.predict(test_img1)
predicted_img2 = predict.predict(test_img2)
print("Prediction complete")

#display both images
cv2.imshow("window1", predicted_img1)
cv2.imshow("window2", predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()