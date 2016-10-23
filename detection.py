import numpy as np
import cv2

# def edges(img):
def person(img, minval, maxval):
	# median = np.median(img)
	# minval = 0.5*median
	# maxval = 1.3*median
	edges = cv2.Canny(img, minval, maxval)
	# edges_show = cv2.resize(edges, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
	# cv2.imshow('canny', edges_show)
	return edges

def thresh(img):
	# median = np.median(img)
	thrsh = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 77, 2)
	return thrsh;

def contour(img):
	#th = thresh(img)
	th = edges(img)
	contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	return contours

# Returns the contours and the b&w mask
# def person(img, cnt_thrsh):
# 	contours2 = []
# 	contours = contour(img)
# 	for cnt in contours:
# 		if(cv2.arcLength(cnt,True) >= cnt_thrsh):
# 			contours2.append(cnt)

# 	mask = np.ones(img.shape[:2], dtype="uint8") * 255
# 	cv2.drawContours(mask,contours2, -1, (0,255,0), 3)

# 	return contours2, mask

