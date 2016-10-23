from __future__ import division	#force / to be float division, // needed for int division
import numpy as np
import cv2
import detection as dt
from matplotlib import pyplot as plt 

'''------------Global Variables----------------'''

window = ' '
img = []
destroy_Window = False
done_Thresholding = False
done_Selecting = False
goto_select = False
drawing = False
ix = -1
iy = -1
fx = -1
fy = -1

'''--------------------------------------------'''

def threshold(x):
	global window, img, person
	minval = cv2.getTrackbarPos('minVal', window)
	maxval = cv2.getTrackbarPos('maxVal', window)
	person = dt.person(img,minval,maxval)
	cv2.imshow(window, person)


def done_thresholding(done):
	global done_Thresholding
	if done == 1:
		threshold(0)
		done_Thresholding = True


def select_area_window():
	global window, img, goto_select
	cv2.destroyWindow(window)
	cv2.namedWindow(window)
	cv2.setMouseCallback(window, select_area)
	cv2.imshow(window,person)
	goto_select = True

def select_area(event, x, y, flags, param):
	global img, window, ix, iy, fx, fy
	global goto_select, done_Selecting, drawing
	temp_img = person.copy()
	rect = np.ones_like(temp_img)

	if goto_select == True:
		if event == cv2.EVENT_LBUTTONDOWN:
			drawing = True
			ix,iy = x,y

		elif event == cv2.EVENT_MOUSEMOVE:
			if drawing == True:
				cv2.rectangle(rect, (ix,iy),(x,y),255,-1)
				temp_img = cv2.addWeighted(temp_img, 1, rect, .3,0)
				cv2.imshow(window, temp_img)

		elif event == cv2.EVENT_LBUTTONUP:
			if drawing == True:
				drawing = False
				cv2.rectangle(temp_img, (ix,iy),(x,y),0,0)
				fx = x
				fy = y
				done_Selecting = True

def find_level_points(level):
	global img, ix, fx, fy
	
	y = 0
	found = 0
	level_points = [0 for i in xrange(2)]
	while level + y <= fy:
		i = ix
		while i <= fx  and found < 1:
			if not person[level + y,i] == 0:
					# take the expected height instead of the real one, for simplicity in calculations
				level_points[found] = (i, level) #level+y if we want to take the exact height 
				found = found + 1
			i = i + 1
		k = fx
		while k > i and found < 2:
			if not person[level + y,k] == 0:
					# take the expected height instead of the real one, for simplicity in calculations
				level_points[found] = (k, level) #level+y if we want to take the exact height 
				found = found + 1
			k = k - 1
		if found == 2:
			return level_points, y
		y = y + 1

def find_silhouette(levels):
	global img, iy, fy
	j = iy
	counter = 0
	silhouette_pairs = [[0 for i in xrange(2)] for i in xrange(levels)]
	jump = (fy-iy)//levels
	while j <= fy and counter < levels:
		print counter
		silhouette_pairs[counter][:], skip = find_level_points(j)
		j = j + jump
		counter = counter + 1
	return silhouette_pairs

def main_loop(image, window_name,levels):
	global window, img
	global destroy_Window, done_Thresholding, done_Selecting, goto_select
	window = window_name
	img = image.copy()
	cv2.namedWindow(window)
	cv2.createTrackbar('minVal', window, 0, 500, threshold)
	cv2.createTrackbar('maxVal', window, 500, 500, threshold)
	cv2.createTrackbar('DONE', window, 0, 1, done_thresholding)

	threshold(0)

	while(1):
		cv2.waitKey(1)
		if  done_Thresholding == True:
			# cv2.destroyWindow(window)
			select_area_window()
			done_Thresholding = False
		if done_Selecting == True:
			cv2.destroyWindow(window)
			return find_silhouette(levels)

'''-----------------FOR TESTING----------------------'''
sample_image = cv2.imread('./testimages5/calPattern0000.jpg',0)
silhouette = main_loop(sample_image,'mainloop',100)
print silhouette
for points in silhouette:
	cv2.circle(sample_image, points[0],1,255,-1)
	cv2.circle(sample_image, points[1],1,255,-1)
cv2.imshow('Found Points',sample_image)

while(1):
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
'''--------------------------------------------------'''