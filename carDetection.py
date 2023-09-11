import cv2
import numpy as np

class CarDetector:

    def __init__(self, image_path):
        self.image_path = image_path

    def detection(self):

        # self.img = cv2.imread(self.image_path)
        # self.img = cv2.resize(self.img, self.img_size)
        img_arr = np.array(self.image_path)

        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        dilated = cv2.dilate(blur, np.ones((3,3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        car_cascade_src = 'detect/car.xml'
        car_cascade = cv2.CascadeClassifier(car_cascade_src)
        cars = car_cascade.detectMultiScale(closing, 1.1, 1)
        return cars