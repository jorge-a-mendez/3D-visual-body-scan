import numpy as np
import cv2
import detection as dt
from matplotlib import pyplot as plt 

drawing = False # true if mouse is pressed
ix,iy = -1,-1
infLine_img = cv2.imread("1InfLine_1.jpg", 0)
# infLine_img = cv2.resize(infLine_img, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
goto_select = False
destroyWindow = False
done_finding = False
HEAD = 0
FEET = 1
person_numbers = ['Center','Left','Right']
head_or_feet = ['HEAD', 'FEET']
destroyed = 0
heads_feet = [[0 for i in xrange(3)] for i in xrange(2)]

def detect_and_overlap(thresh, window='people0'):
	i = 0
	person = []
	minval = cv2.getTrackbarPos('minVal', window)
	maxval = cv2.getTrackbarPos('maxVal', window)	
	for f in img:
	    

	    temperson = dt.person(f, minval, maxval)
	    # print temperson


	    # temperson = cv2.resize(temperson, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
	    person.append(temperson)
	    # cv2.imshow("contourcitos",person[i])
	    # cv2.waitKey(1000)

	    i = i + 1	

	return display_overlapped(person,window)

def display_overlapped(person, window):
	overlapped = person[0] | person[1] | person[2]
	cv2.imshow(window,overlapped)
	return overlapped

def done_thresholding(done):
	global infLine_img, goto_select, destroyWindow
	if done == 1:
		window = 'people%d' % (destroyed)
		infLine_img = detect_and_overlap(0)
		goto_select = True
		select_area_window()
		
def select_area_window(head_feet = HEAD, person_number = 0):
	global destroyed, destroyWindow, infLine_img
	img = infLine_img.copy()
	window = 'people%d' % (destroyed +1)
	cv2.namedWindow(window)
	cv2.setMouseCallback(window, select_area,(window,head_feet,person_number))
	text = head_or_feet[head_feet] + ' ' + person_numbers[person_number]
	((width, height),baseline) = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX, 1,1)
	cv2.putText(img,text,(0,height),cv2.FONT_HERSHEY_SIMPLEX, 1,255,thickness=1)
	cv2.imshow(window,img)
	destroyWindow = True

# def select_area_window(window,head_feet,person_number):
# 	cv2.destroyWindow('people')
# 	cv2.namedWindow('people')
# 	cv2.imshow('people',infLine_img)
# 	cv2.setMouseCallback('people', select_area)
# 	goto_select = True

def find_point(img,(ix,iy),(x,y),head_feet,person_number):
	# ROI = img[iy:y, ix:x]
	# cv2.imshow('ROI',ROI)
	# print ROI

	if head_feet == HEAD:
		i = ix
		j = iy
		while j < y:
			while i < x:
				if not img[j,i] == 0:
					return (i,j)
				i = i + 1
			j = j + 1
			i = ix
	elif head_feet == FEET:
		i = x
		j = y
		while j > iy:
			while i > ix:
				if not img[j,i] == 0:
					return (i,j)
				i = i - 1
			j = j - 1
			i = x	

def select_next(head_feet, person_number):
	global destroyWindow, done_finding, goto_select
	if person_number == 2:
		head_feet = head_feet + 1
	person_number = (person_number + 1) % 3
	if head_feet > 1:
		# destroyWindow = True
		done_finding = True
		goto_select = False
	else:
		select_area_window(head_feet,person_number)

def select_area(event, x, y, flags, (window,head_feet,person_number)):
	# head_feet = param[0]
	# person_number = param[1]
	global infLine_img, drawing, ix, iy
	temp_img = infLine_img #infLine_img[:]
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
			drawing = False
			cv2.rectangle(temp_img, (ix,iy),(x,y),0,0)
			# temp_img = cv2.addWeighted(temp_img, 1, rect, .3,0)
			
			point = find_point(temp_img,(ix,iy),(x,y),head_feet,person_number)
			cv2.circle(temp_img, point,5,255,-1)
			cv2.imshow(window, temp_img)
			heads_feet[head_feet][person_number] = point
			print heads_feet
			select_next(head_feet,person_number)
			
def inf_line():
	print 'Calculate line at infinity!'
	# Move feet points to exactly under the corresponding head
	# Compute the lines that join:
		# Feet 0 with Feet 1 -> Line 1
		# Head 0 with Head 1 -> Line 2
		# Feet 0 with Feet 2 -> Line 3
		# Head 0 with Head 2 -> Line 4
	# Obtain the intersection points between
		# Line 1 and Line 2 -> VP1
		# Line 3 and Line 4 -> VP2
	# Obtain the line at the infinity as the line that joins VP1 and VP2

img = []

i = 0
while(i < 3):

	title = "1InfLine_%d.jpg" %(i)
	gray = cv2.imread(title, 0)
	gray = cv2.resize(gray, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
	img.append(gray)

	i = i + 1

init_thresh = 70

window = 'people%d' % (destroyed)
cv2.namedWindow(window)
# cv2.createTrackbar('Threshold',window,init_thresh,300,detect_and_overlap)
cv2.createTrackbar('minVal', window, 0, 500,detect_and_overlap)
cv2.createTrackbar('maxVal', window, 500, 500, detect_and_overlap)
cv2.createTrackbar('DONE', window,0,1,done_thresholding)


detect_and_overlap(window)


while(1):
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
	if destroyWindow == True:
		window = 'people%d' % (destroyed)
		cv2.destroyWindow(window)
		destroyed = destroyed + 1
		destroyWindow = False
	if done_finding == True:
		inf_line()
		done_finding = False

cv2.destroyAllWindows()
