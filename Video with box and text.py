'''
Loading a save video
Create a rectangle
Put a text in it

'''

import numpy as np
import cv2

cap = cv2.VideoCapture('D:/DataScience/OpenCV/faces.mp4')
i= 0
while(cap.isOpened()):
    ret, frame = cap.read()
    text = "Count = " + str(i)

    cv2.rectangle(frame,(500,0),(640,50),(255,255,255),2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,text,(520,30), font, 0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(43) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()