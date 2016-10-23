import numpy as np
import cv2
import detection as dt
from matplotlib import pyplot as plt 
import sys

if len(sys.argv) < 3:
    print "Error!\n\t python videoopen.py [relative path to video] [relative path where to save the images]\n"
else: 

    cap = cv2.VideoCapture(sys.argv[1])
    i = 0

    ret, frame = cap.read()
    while(cap.isOpened() and ret):

        # Our operations on the frame come here
        frame = cv2.transpose(frame)        # rotate..
        frame = cv2.flip(frame,1)           # ..90 degrees
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        title = "/img%.4d.jpg" % (i)
        title = sys.argv[2] + title
        frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        cv2.imwrite(title, frame)

        i = i + 1

        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    	# Capture frame-by-frame
        ret, frame = cap.read()



    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
