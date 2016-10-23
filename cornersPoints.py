import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
i = 0;
while i <= 100:
	filename = "testimages4/calPattern%.4d.jpg" % i;
	f = cv2.imread(filename);
	f = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY);
	f1 = np.float32(f);
	dst = cv2.cornerHarris(f1,2,3,0.04);
	dst = cv2.dilate(dst,None);
	f[dst > 0.01*dst.max()] = 0;
	i = i + 1;
	cv2.imshow('image',f);
	cv2.waitKey(10);
