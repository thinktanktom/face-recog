import detect_face
import draw_rectangle
def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face.detect_face(img)
    draw_rectangle.draw_rectangle(img, rect)
    return img