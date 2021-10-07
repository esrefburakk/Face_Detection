import cv2
import numpy as np

cap = cv2.VideoCapture("faces.mp4")
face_cascade = cv2.CascadeClassifier("frontalface.xml")

while True:
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_cascade_gray = face_cascade.detectMultiScale(frame_gray,1.1,2)

    for x,y,w,h in face_cascade_gray:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("Image",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
