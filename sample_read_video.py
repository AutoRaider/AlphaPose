import numpy as np
import cv2
import os


origin = os.getcwd()
video_path = os.path.join(origin, 'badminton.mp4')

cap = cv2.VideoCapture(video_path)
i = 1

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
    	break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    cv2.waitKey(10)
    print(i)
    i=i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()