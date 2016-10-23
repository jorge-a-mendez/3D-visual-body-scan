import cv2, cv
import numpy as np
import sys
# arg_1: phone name
# arg_2: number of patterns to use


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

patterns = int(sys.argv[2])

i = 0
j = 0

video = sys.argv[1] + ".mp4"

cap = cv2.VideoCapture(video)

frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
jump = int(frames/(patterns + 2))		# +1 to leave some room for not finding one


ret, frame = cap.read()
while(cap.isOpened() and ret):
    
    print "Frame: %d" % (i)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)       #resize to third the size.
    # frame = cv2.transpose(frame)        # rotate..
    # frame = cv2.flip(frame,1)   
    # Find the chess board corners
    foundPattern, corners = cv2.findChessboardCorners(frame, (8,6),None)

    # If found, add object points, image points (after refining them)
    
    if foundPattern:
    	j = j + 1
        objpoints.append(objp)

        cv2.cornerSubPix(frame,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, (8,6), corners,foundPattern)
        cv2.namedWindow("img", cv2.CV_WINDOW_AUTOSIZE)
        frame = cv2.resize(frame, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
        cv2.imshow("img",frame)
        cv2.waitKey(500)
        i = i + jump
        for k in range(jump):
        	cap.read()

    if j >= patterns:
    	break
    
    print "Patterns: %d\n\n" % (j)

    i = i + 1

    ret, frame = cap.read()

    
cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(video)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print "calibrating..."
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame.shape[::-1],None,None)

aux = mtx[0][2]
mtx[0][2] = mtx[1][2]
mtx[1][2] =  aux
mtx = mtx * 0.3
mtx[2][2] = 1

print mtx
print "retval = %d" %(ret)

filename = sys.argv[1] + "_cal.txt"

np.savetxt(filename, mtx)

cap.release()
cv2.destroyAllWindows()