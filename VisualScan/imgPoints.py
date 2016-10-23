''' ================================================================================================
		This module contains the methods to compute all the information related to the silhouette 
		points needed to approximate the contour with a visual hull approach.
    ================================================================================================ '''

from __future__ import division	#force / to be float division, // needed for int division
import numpy as np
import cv2
import detection as dt
from matplotlib import pyplot as plt 
import worldInfo as wi
from constants import * 


#---------GLOBAL VARIABLES FOR CLASS-CALLBACKS INTERFACING----------#
goto_refresh = False
goto_done = False
#--------------------------------------------------------------------#

# Public methods:
#	select_threshold(img): selects the canny thresholds
#	select_area(distance,img): selects the area of the person to find edges
#							receives the dstance of the person in current frame
#	get_silhouette(distance,img): returns the img points correspongind
#							to edges of the person, quantized in 
#							'levels' levels;receives the distance of person current
#							 frame
# 	set_origDistance(distance): set distance of image used for select_area
#	get_headPoint(distance,img)


class imgPoints:
	def __init__(self, levels=100):
		self.levels = levels

	def select_threshold(self, img):
		global goto_refresh, goto_done
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		self.window = 'Thresholding'
		cv2.namedWindow(self.window)
		cv2.createTrackbar('minVal',self.window,0,1000,goto_refreshEdges)
		cv2.createTrackbar('maxVal',self.window,1000,1000,goto_refreshEdges)
		cv2.createTrackbar('DONE',self.window,0,1,goto_doneThresholding)
		edges = self.__refreshEdges__(img)
		cv2.imshow(self.window,edges)
		self.done_thresholding = False
		self.thresholds = [0,500]
		while(1):
			cv2.waitKey(1)
			if goto_refresh == True:
				edges = self.__refreshEdges__(img)
				cv2.imshow(self.window,edges)
				goto_refresh = False
			if goto_done == True:
				edges = self.__refreshEdges__(img)
				cv2.destroyWindow(self.window)
				cv2.namedWindow(self.window)
				cv2.imshow(self.window,edges)
				cv2.waitKey(500)
				cv2.destroyWindow(self.window)
				a = np.array(self.thresholds)
				np.savetxt("thresholds.txt",a)
				break

		# Uncomment the following lines to load threshold values from file for debugging
		# a = np.loadtxt("thresholds.txt")
		# a = tuple(a)
		# self.thresholds = a

	# Use this method instead of select_area to maunually select the ROI
	# def select_area_MANUAL(self,img):
	# 	self.done_selecting = False
	# 	self.drawing = False
	# 	self.window = 'SelectArea'
	# 	cv2.namedWindow(self.window)
	# 	edges = self.__edges__(img)
	# 	cv2.setMouseCallback(self.window,callback_select,(self,edges))
	# 	cv2.imshow(self.window, edges)
	# 	self.orig_ix, self.orig_iy = -1,-1
	# 	self.orig_fx, self.orig_fy = -1,-1
	# 	while(1):
	# 		cv2.waitKey(1)
	# 		if self.done_selecting == True:
	# 			edges = self.__edges__(img)
	# 			cv2.rectangle(edges, (self.orig_ix,self.orig_iy),(self.orig_fx,self.orig_fy),255,0)
	# 			cv2.imshow(self.window,edges)
	# 			cv2.waitKey(2000)
	# 			cv2.destroyWindow(self.window)
	# 			break

	def select_area(self,img, world):
		self.orig_ix = img.shape[1]*NUM_ORIG_IX//DEN_ORIG_IX
		self.orig_fx = img.shape[1]*NUM_ORIG_FX//DEN_ORIG_FX
		self.orig_iy = img.shape[0]//2	# look for head from 1/5 to 4/5 in x, 1/10 to 1/2 in y
		headpoint = self.get_headPoint(img)
		self.orig_iy = world.get_imgHeightAtHeadDistance(headpoint,world.person_height*NUM_ORIG_IY/DEN_ORIG_IY)
		self.orig_fy = world.get_imgHeightAtHeadDistance(headpoint,world.person_height*NUM_ORIG_FY/DEN_ORIG_FY)

	def set_origDistanc(self, distance):
		self.orig_distance = distance

	def get_silhouette(self, new_iy,new_fy, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ix = int(self.orig_ix)
		iy = int(new_iy)
		fx = int(self.orig_fx)
		fy = int(new_fy)
		edges = self.__edges__(img)
		j = iy
		counter = 0
		levels = self.levels
		silhouette_pairs = [[0 for i in xrange(2)] for i in xrange(levels)]
		jump = (fy-iy)/levels
		while j <= fy and counter < levels:
			ret = self.__find_level_points__(edges,ix,int(j),fx,fy)
			if ret == None:
				return None
			level_points = ret
			silhouette_pairs[counter][:] = level_points
			j = j + jump
			counter = counter + 1
		return silhouette_pairs


	def get_headPoint(self, img, distance=None,use_distance=False):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		edges = self.__edges__(img)

		ix,iy = self.orig_ix, self.orig_iy
		fx = self.orig_fx

		j = img.shape[0]*NUM_BORDER_HEAD//DEN_BORDER_HEAD		# ignore 10% border area 

		while j < iy:
			i = ix
			while i < fx:
				if edges[j,i] != 0:
					return (i,j)
				i = i + 1
			j = j + 1

		return None


	def __edges__(self, img):
		return cv2.Canny(img, self.thresholds[0],self.thresholds[1])

	def __find_level_points__(self,img,ix,level,fx,fy):

		y = 0
		found = 0
		level_points = [0 for i in xrange(2)]
		# while level + y <= fy:
		if found < 1:
			i = ix
			while i <= fx  and found < 1:
				if not img[level,i] == 0:
						# take the expected height instead of the real one, for simplicity in calculations
					level_points[found] = (i, level) #level+y if we want to take the exact height 
					found = found + 1
				i = i + 1

		k = fx
		while k > i+5 and found < 2:
			if not img[level,k] == 0:
					# take the expected height instead of the real one, for simplicity in calculations
				level_points[found] = (k, level) #level+y if we want to take the exact height 
				found = found + 1
			k = k - 1
		if found == 2:
			# print level_points
			return level_points
		return None

	def __refreshEdges__(self, img):
		minval = cv2.getTrackbarPos('minVal',self.window)
		maxval = cv2.getTrackbarPos('maxVal',self.window)
		self.thresholds = [minval, maxval]
		return self.__edges__(img)

#------------------CALLBACKS FOR THE CLASS WORLDINFO-------------------#
def callback_select(event, x, y, flags, (img_points,edges)):
	temp_img = edges.copy()
	rect = np.ones_like(temp_img)
	if event == cv2.EVENT_LBUTTONDOWN:
		img_points.drawing = True
		img_points.orig_ix,img_points.orig_iy = x,y

	elif event == cv2.EVENT_MOUSEMOVE:
		if img_points.drawing == True:
			cv2.rectangle(rect, (img_points.orig_ix,img_points.orig_iy),(x,y),255,-1)
			temp_img = cv2.addWeighted(temp_img, 1, rect, .3,0)
			cv2.imshow(img_points.window, temp_img)

	elif event == cv2.EVENT_LBUTTONUP:
		if img_points.drawing == True:
			img_points.drawing = False
			img_points.orig_fx, img_points.orig_fy = x,y
			img_points.done_selecting = True

def goto_refreshEdges(x):
	global goto_refresh
	goto_refresh = True

def goto_doneThresholding(done):
	global goto_done
	if done == 1:
		goto_done = True
#---------------------------------------------------------------------#


'''-----------------FOR TESTING----------------------'''
# wi_img_names = 'InfLine_'
# phone = 'OnePlus'
# height = 180
# rotating_img_names = '/testimages5/calPattern'

# wi_img = []
# i = 0
# while(i < 3):	
# 	# print i
# 	title = wi_img_names + "%d.jpg" %(i)
# 	gray = cv2.imread(title, 0)
# 	# print gray
# 	# gray = cv2.resize(gray, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
# 	wi_img.append(gray)
# 	i = i + 1

# title = phone + '_cal.txt'
# kmatrix = np.loadtxt(title)
# world = wi.worldInfo(wi_img,height,kmatrix)
# world.select_heads_feet()  # this calculates plane at infty and stores it in world.headsplany
# 							# it also makes it able to compute distance to person.
# 							# probably should also compute 3D points completely, for intuitiveness


# imgpoints = imgPoints(50)
# select_img = cv2.imread('./testimages6/calPattern0000.jpg',0)
# random_img = cv2.imread('./testimages6/calPattern0200.jpg',0)
# imgpoints.select_threshold(select_img)
# # a = np.array(imgpoints.thresholds)
# # np.savetxt("thresholds.txt",a)
# imgpoints.select_area(select_img,world)

# headpoint = imgpoints.get_headPoint(select_img,use_distance=False)
# orig_HeadToNWCorner = world.get_head2PointDistance(headpoint,[imgpoints.orig_ix,imgpoints.orig_iy])
# orig_HeadToSWCorner = world.get_head2PointDistance(headpoint,[imgpoints.orig_ix,imgpoints.orig_fy])

# head_top_img = imgpoints.get_headPoint(random_img,use_distance=False)
# new_iy = world.get_imgHeightAtHeadDistance(head_top_img,orig_HeadToNWCorner)
# new_fy = world.get_imgHeightAtHeadDistance(head_top_img,orig_HeadToSWCorner)

# silhouette = imgpoints.get_silhouette(new_iy, new_fy, random_img)
# # print 'Silhouette'
# # print silhouette
# for points in silhouette:
# 	if points[0] != 0:
# 		cv2.circle(random_img, points[0],1,255,-1)
# 		cv2.circle(random_img, points[1],1,255,-1)
# cv2.imshow('Found Points',random_img)


# while(1):
# 	k = cv2.waitKey(0) & 0xFF
# 	if k == 27:
# 		break

# cv2.destroyAllWindows()
'''--------------------------------------------------'''