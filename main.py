import cv2
import numpy as np

print("pkg imported")

img=cv2.imread("resources/lena.png")
img2=cv2.imread("resources/Soorya.jpg")

print(img2.shape)

cv2.waitKey(0)