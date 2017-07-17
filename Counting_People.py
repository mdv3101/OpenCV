'''
Detecting Number of Faces by using Haar-Cascade
'''

import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier("C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml")
UpperCascade = cv2.CascadeClassifier("C:/opencv/build/etc/haarcascades/haarcascade_upperbody.xml")
LowerCascade = cv2.CascadeClassifier("C:/opencv/build/etc/haarcascades/haarcascade_lowerbody.xml")
FullCascade = cv2.CascadeClassifier("C:/opencv/build/etc/haarcascades/haarcascade_fullbody.xml")
cap = cv2.VideoCapture('D:/DataScience/OpenCV/faces.mp4')
i= 0
#Background Subtraction is not suitable, since background is dynamic
#fgbg = cv2.createBackgroundSubtractorMOG2()

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    '''
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = fgmask[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    '''
    
    Upper = UpperCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    '''
    for (x,y,w,h) in Upper:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(148,255,0),2)
        roi_gray = fgmask[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    '''
    Lower = LowerCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    '''
    for (x,y,w,h) in Lower:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(148,255,148),2)
        roi_gray = fgmask[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    '''
    #The full cascade is of minimal use in the given video faces.mp4
    '''
    Full = FullCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
        
    )
    '''
    '''
    for (x,y,w,h) in Full:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
        roi_gray = fgmask[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    '''
    j=0
    #Detecting overlapping of FaceCascade and UpperCascade
    for (x1,y1,w1,h1) in faces:
        for (x2,y2,w2,h2) in Upper:
            if x2<=x1 and y2<=y1 and (x2+w2)>=(x1+w1) and (x2+h2)>=(x1+h1):
                j=1

    
    a= len(faces)
    b= len(Upper)
    c= len(Lower)
    #d= len(Full) 
    i=a+b+c-j
    text = "Count = " + str(i)
    
    #display a white rectangle in top right corner
    cv2.rectangle(frame,(500,0),(640,50),(255,255,255),2)
    
    #display count in top right corner
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    cv2.putText(frame,text,(520,30), font, 0.7,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('Output',frame)
    
    #waitKey is set to 1 for 1msec wait time between frames. 
    #The video can be terminated by using 'q' key on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()