import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

im = cv2.imread("./testimages4/calPattern0180.jpg")
im2 = cv2.imread("./testimages4/calPattern0190.jpg")
if im == None:
	print "Couldn't open file\n"
else:
	sift = cv2.xfeatures2d.SIFT();
	kp,des = sift.detectAndCompute(im,None);
	kp2,des2 = sift.detectAndCompute(im2,None);
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des,des2,k=2)
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m]);
	img = cv2.drawMatches(im,kp,im2,kp2,good,flags=2)
	plt.imshow(img)
	plt.show()