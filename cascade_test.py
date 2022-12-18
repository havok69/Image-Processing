import cv2
import numpy as np

test_cascade = cv2.CascadeClassifier("Resources/cascade.xml")

img = cv2.imread("Resources/test3.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

obj = test_cascade.detectMultiScale(gray,1.01,7)
for (x,y,w,h) in obj:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()