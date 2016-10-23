import cv2,cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import itemfreq
import sys

#open a window to select the color of the face to track
#keeps track of the change of rotation between frames.
#should be able to return the change of orientation if it was successfully updated 
#should have methods to update the orientation and get the last orientation computed.
#Still needed to decide if it should be used edges and lines or the contour of the segmented area.


class boxFaceTracker:
	def __init__(self,frame,id,CCW = True, faces = 1): 
		
		self.point = ()
		self.newpoint = False
		self.frames = 0
		self.faces = faces
		self.sense = CCW
		self.id = id
		self.rotation = 0
		self.alpha = 0.04
		self.img = frame
		# get the color to segment
		# cv2.namedWindow("SelectColors%s"%id)
		# cv2.setMouseCallback('SelectColors%s'%id,self.__select_color__)
		# cv2.imshow("SelectColors%s"%id,frame)
		
		# while not self.newpoint:
		# 	k = cv2.waitKey(1) & 0xFF
		# 	if k == 27:
		# 		break
		# cv2.destroyWindow('SelectColors%s'%id)
		# self.newpoint = False
		# self.hsv = hsv[self.point[1],self.point[0]];
		hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		

		self.hue = [0,26]
		self.sat = [137,255]
		self.value = [158,255]
		window = "Thresholds"
		cv2.namedWindow(window)
		cv2.createTrackbar('Hue min', window, self.hue[0],179, self.__getLimit__)
		cv2.createTrackbar('Hue max', window, self.hue[1],179, self.__getLimit__)
		cv2.createTrackbar('Sat min', window, self.sat[0],255, self.__getLimit__)
		cv2.createTrackbar('Sat max', window, self.sat[1],255, self.__getLimit__)
		cv2.createTrackbar('Value min', window, self.value[0],255, self.__getLimit__)
		cv2.createTrackbar('Value max', window, self.value[1],255, self.__getLimit__)
		
		while cv2.waitKey(1) & 0xFF != 27:
			# self.__segmentNgetCorners__(hsv);
			lowerbound = (self.hue[0],self.sat[0],self.value[0])
			upperbound = (self.hue[1],self.sat[1],self.value[1])
			mask = cv2.inRange(hsv,lowerbound,upperbound)
			cv2.imshow('mask',mask)
			# cv2.waitKey(1)
		cv2.destroyAllWindows()

	def __getLimit__(self,thresh,window ="Thresholds" ):
		self.hue = []
		self.sat = []
		self.value = []
		self.hue.append(cv2.getTrackbarPos('Hue min',window))
		self.hue.append(cv2.getTrackbarPos('Hue max',window))
		self.sat.append(cv2.getTrackbarPos('Sat min',window))
		self.sat.append(cv2.getTrackbarPos('Sat max',window))
		self.value.append(cv2.getTrackbarPos('Value min',window))
		self.value.append(cv2.getTrackbarPos('Value max',window))

	def __segmentNgetCorners__(self,imghsv):
		# htl = 5
		# stl = 255
		# vtl = 255
		# htu = 5
		# stu = 255
		# vtu = 255
		# if (self.hsv[0]-htl < 0):
		# 	h = 0
		# 	htu = htu + htl - self.hsv[0]
		# else:
		# 	h = self.hsv[0] - htl
		# if (self.hsv[1]-stl < 0):
		# 	s = 100
		# 	stu = stu + stl - self.hsv[1]
		# else:
		# 	s = self.hsv[1] - stl
		# if (self.hsv[2]-vtl < 0):
		# 	v = 100
		# 	vtu = vtu + vtl - self.hsv[2]
		# else:
		# 	v = self.hsv[2] - vtl

		# lowerbound = (h,s,v)
		# print lowerbound
		# if (self.hsv[0]+htu > 179):
		# 	h = 179
		# else:
		# 	h = self.hsv[0] + htu
		# if (self.hsv[1]+stu > 255):
		# 	s = 255
		# else:
		# 	s = self.hsv[1] + stu
		# if (self.hsv[2]+vtu > 255):
		# 	v = 255
		# else:
		# 	v = self.hsv[2] + vtu
		
		# upperbound = (h,s,v)
		# print upperbound

		lowerbound = (self.hue[0],self.sat[0],self.value[0])
		upperbound = (self.hue[1],self.sat[1],self.value[1])
		print lowerbound
		print upperbound
		mask = cv2.inRange(imghsv,lowerbound,upperbound)
		
		kernel = np.ones((3,3),np.uint8)
		# mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
		mask = cv2.dilate(mask,None,iterations = 2)	
		
		# update the average color of the region
		# print self.hsv
		# npoints = np.nonzero(mask == 255)[0].size
		# if npoints != 0:
		# 	self.hsv = (1-self.alpha)*self.hsv + self.alpha*sum(np.uint32(imghsv[mask == 255]))/npoints
		# print self.hsv

		h,s,bw = cv2.split(imghsv)
		edges = cv2.Canny(bw,10,100)
		edges[mask == 0] = 0
		cv2.imshow('edges',edges)
		cv2.imshow('mask',mask)
		# cv2.waitKey(0)

		# lines = cv2.HoughLines(edges,10,10*np.pi/180,10)
		# for rho,theta in lines[0]:
		# 	a = np.cos(theta)
		# 	b = np.sin(theta)
		# 	x0 = a*rho
		# 	y0 = b*rho
		# 	x1 = int(x0 + 1000*(-b))
		# 	y1 = int(y0 + 1000*(a))
		# 	x2 = int(x0 - 1000*(-b))
		# 	y2 = int(y0 - 1000*(a))

		# 	cv2.line(imghsv,(x1,y1),(x2,y2),(0,0,255),2)
		# cv2.imshow('lines',imghsv)
		# cv2.waitKey(1)

		contours, level = cv2.findContours(edges,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
		i = 0
		while i >= 0 and level != None:
			cv2.drawContours(img,contours,i,(0,255-i*10,i*30),1)
			i = level[0,i,0]
		cv2.drawContours(img,contours,-1,(0,0,0),1)
		approxPoly = None 
		if contours != None:
			for j in contours:
				approxPoly = cv2.approxPolyDP(j,5,True)
				approxPoly = approxPoly[:,0,:]
			if approxPoly != None:
				# for k in list(approxPoly):
				# 	cv2.circle(img,(k[0],k[1]),2,(0,0,255),-1)
				cv2.imshow('corners',img)
				# cv2.waitKey(1)
			else:
				print 'Polygon not found '
		else:
			print 'contours not found'	

		dst = cv2.cornerHarris(edges,2,3,0.15)
		#result is dilated for marking the corners, not important
		# dst = cv2.dilate(dst,None)
		# Threshold for an optimal value, it may vary depending on the image.
		bordersp = np.zeros(mask.shape,np.uint8)
		bordersp[dst>0.2*dst.max()]=255
		img[dst>0.2*dst.max()]=(0,255,255)
		points = np.nonzero(dst > 0.2*dst.max())
		points = np.vstack([points[1],points[0]])
		print 'points'
		print points
		centroid = np.int32(np.mean(points,axis = 1))
		print centroid
		cv2.circle(img,(centroid[0],centroid[1]),2,(255,255,0))
		if points.size != 0:
			for l in range(points.shape[1]):
				points[:,l] = points[:,l] - centroid.transpose()

			u,s,v = np.linalg.svd(points)
			s = np.uint32(s)
			print s
			angle = np.degrees(np.arctan2(u[1,0],u[0,0]));
			print angle 
			cv2.ellipse(img,(centroid[0],centroid[1]),(s[0]/2,s[1]/2),angle,0,360,255,0)


		# img[centroid[0],centroid[1]] = (255,0,0)
		# approxPoly = cv2.approxPolyDP(points,7,True)
		# approxPoly = approxPoly[:,0,:]
		# if approxPoly != None:
		# 	for k in list(approxPoly):
		# 		cv2.circle(img,(k[0],k[1]),1,(0,255,0),-1)
		# 	cv2.imshow('corners',img)
		# 	# cv2.waitKey(1)
		# else:
		# 	print 'Polygon not found '
		regions,markers = self.__classify__(mask)
		img[markers == -1] = (255,0,0)
		# hull = cv2.convexHull(points)
		# for k in list(hull):
		# 	k = k[0]
		# 	cv2.circle(img,(k[0],k[1]),1,(0,255,0),-1)
		cv2.imshow('corners',img)
		cv2.waitKey(5)

		# lines = cv2.HoughLines(bordersp,5,1*np.pi/180,20)
		# for rho,theta in lines[0]:
		# 	a = np.cos(theta)
		# 	b = np.sin(theta)
		# 	x0 = a*rho
		# 	y0 = b*rho
		# 	x1 = int(x0 + 1000*(-b))
		# 	y1 = int(y0 + 1000*(a))
		# 	x2 = int(x0 - 1000*(-b))
		# 	y2 = int(y0 - 1000*(a))

		# 	cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
		# cv2.imshow('lines',img)
		# cv2.imshow('pointsforhough',bordersp)
		cv2.waitKey(0)

		return points

	def __select_color__(self,event, x, y, flags, params):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.point = (x,y)
			self.newpoint = True

	def detect(self,newframe):
		hsv = cv2.cvtColor(newframe,cv2.COLOR_BGR2HSV)
		corners = self.__segmentNgetCorners__(hsv)
		# self.__RatioOfLengths__(corners)

	# def __RatioOfLengths__(self,corners):

	def __classify__(self, markers):
		# noise removal 
		kernel = np.ones((3,3),np.uint8)
		opening = cv2.morphologyEx(markers,cv2.MORPH_OPEN,kernel,iterations=3)

		#sure bg
		surebg = cv2.dilate(opening,kernel,iterations = 1)

		# sure foreground
		distTransform = cv2.distanceTransform(opening,cv.CV_DIST_L2,0)
		ret,surefg = cv2.threshold(distTransform,0*distTransform.max(),255,0)

		#
		surefg = np.uint8(surefg)
		unknown = cv2.subtract(surebg,surefg)

		# marker labelling
		contours, level = cv2.findContours(opening,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
		markers = np.zeros(surefg.shape,np.int32)
		i = 0
		compCount = 0
		while (i >=0 and level != None):
			cv2.drawContours(markers,contours,i,compCount+1,-1,8,level)
			i = level[0,i,0]
			compCount = compCount + 1
		# markers = markers+1
		# markers[unknown == 255] = 0
		cv2.watershed(img,markers)
		region = list(np.unique(markers))
		region.pop(0)
		l = []
		for j in region:
			mask = np.zeros(markers.shape)
			mask[markers == j] = 255
			l.append((mask,j))
		return l,markers


def select_colors(event, x, y, flags, params):
	global face_number,hsv_colors, holding,hsv
	if event == cv2.EVENT_LBUTTONDOWN:
		if not holding:
			(h,s,v) = hsv[y,x]
			hsv_colors.append((h,s,v))
			face_number = face_number + 1
		holding = True
	if event == cv2.EVENT_LBUTTONUP:
		holding = False
	



# #------------------------GLOBAL VARIABLES------------------------#
# face_numbers = ['BOX1FRONT', 'BOX1SIDE', 'BOX2FRONT', 'BOX2SIDE']
# face_number = 0
# hsv_colors = []
# holding = False
# #----------------------------------------------------------------#


# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# cv2.namedWindow("SelectColors")
# cv2.setMouseCallback('SelectColors',select_colors)
# while face_number < 4:
# 	k = cv2.waitKey(1) & 0xFF
# 	if k == 27:
# 		break

# 	show = img.copy()
# 	if face_number < 4:
# 		text = face_numbers[face_number]
# 	((width, height),baseline) = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX, 1,1)
# 	cv2.putText(show,text,(0,height),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),thickness=1)
# 	cv2.imshow('SelectColors',show)

# cv2.destroyWindow('SelectColors')
# print hsv_colors

# k = 30;
# while k <= 350:
# 	filename = "testimages5/calPattern%.4d.jpg" % k;
# 	img = cv2.imread(filename);
# 	f = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# 	edges = cv2.Canny(f,100,50)

	# ====================== Color based segmentation =====================
	

i = 117
img = cv2.imread("testimages5/calPattern%.4d.jpg" % i)
img = cv2.GaussianBlur(img,(5,5),1) 
tracker = boxFaceTracker(img,'Orange')
i = 118
while i <= 300:
	img = cv2.imread("testimages5/calPattern%.4d.jpg" % i)
	img = cv2.GaussianBlur(img,(5,5),0)
	edges = cv2.Canny(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),10,100)
	cv2.imshow('whole edges',edges)
	tracker.detect(img)
	i = i + 1