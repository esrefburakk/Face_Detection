import cv2
import numpy as np

img = cv2.imread("face.png")
face_cascade = cv2.CascadeClassifier("frontalface.xml")

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_cascade_gray = face_cascade.detectMultiScale(img_gray,1.3,5)

for x,y,w,h in face_cascade_gray:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()