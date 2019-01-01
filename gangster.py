import numpy as np
import cv2
import os
import datetime
from utils.utils import image_resize

face_cascade        = cv2.CascadeClassifier('./utils/haarcascade_frontalface_default.xml')
eyes_cascade        = cv2.CascadeClassifier('./utils/frontalEyes35x16.xml')
nose_cascade        = cv2.CascadeClassifier('./utils/Nose18x15.xml')
glasses             = cv2.imread("./images/glasses.png", -1)
hat                 = cv2.imread("./images/hat.png", -1)
mustache            = cv2.imread('./images/mustache.png',-1)


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,960)
ret,frame = cap.read()
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, flipCode=1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    image_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
    if faces!=():
        for (x,y,w,h) in faces:
            cv2.rectangle(gray,(x,y),(x+w,y+h),(255,255,0),thickness=2)

            eyes = eyes_cascade.detectMultiScale(gray[y:y+h, x:x+w], scaleFactor=1.5, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(gray[y:y+h, x:x+h], (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
                # glasses2 = image_resize(glasses.copy(), width=ew)
                # # gray[y:y+h, x:x+h][ey: ey + eh, ex: ex + ew] = cv2.addWeighted(gray[y:y+h, x:x+h][ey: ey + eh, ex: ex + ew],0.5,glasses2,0.5,0)
                # gw, gh, gc = glasses2.shape
                # for i in range(0, gw):
                #     for j in range(0, gh):
                #         #print(glasses[i, j]) #RGBA
                #         if glasses2[i, j][3] != 0: # alpha 0
                #             gray[y:y+h, x:x+h][ey + i, ex + j] = glasses2[i, j]


            nose = nose_cascade.detectMultiScale(gray[y:y+h, x:x+w], scaleFactor=1.5, minNeighbors=5)
            for (nx, ny, nw, nh) in nose:
                cv2.rectangle(gray[y:y+h, x:x+h], (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
                # mustache2 = image_resize(mustache.copy(), width=nw)
                # # gray[y:y+h, x:x+h][ny: ny + nh, nx: nx + nw] = cv2.addWeighted(gray[y:y+h, x:x+h][ny: ny + nh, nx: nx + nw],0.5,mustache2,0.5,0)
                # mw, mh, mc = mustache2.shape
                # for i in range(0, mw):
                #     for j in range(0, mh):
                #         #print(glasses[i, j]) #RGBA
                #         if mustache2[i, j][3] != 0: # alpha 0
                #             gray[y:y+h, x:x+h][ny + int(nh/2.0) + i, nx + j] = mustache2[i, j]
        cv2.imshow('Filter', gray)
    else:
        cv2.imshow('Filter', frame)
