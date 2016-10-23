import cv2,cv
import numpy as np
from matplotlib import pyplot as plt
# from scipy.stats import itemfreq
import sys
from operator import itemgetter
from constants import *
import os

#open a window to select the color of the face to track
#keep track of the change of rotation between frames.


class stickTracker:
	def __init__(self,frame,id,CCW = True): 
		
		self.point = ()
		self.newpoint = False
		self.frames = 0
		self.sense = CCW
		self.id = id
		self.rotation = 0
		self.img = frame
		self.centroid = 0
		self.frames = {}		# symbol table to avoid reanalyzing frames 
		hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		self.rmax = (0,0)
		self.areamax = 0

		cv2.namedWindow("SelectColors%s"%id)
		cv2.setMouseCallback('SelectColors%s'%id,self.__select_color__)
		cv2.imshow("SelectColors%s"%id,frame)
		
		while not self.newpoint:
			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				break
		cv2.destroyWindow('SelectColors%s'%id)
		self.newpoint = False
		self.hsv = hsv[self.point[1],self.point[0]];
		self.centroid = self.point

		self.hue = [self.hsv[0]-5,self.hsv[0]+5]
		self.sat = [0,255]
		self.value = [0,255]
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

	def __select_color__(self,event, x, y, flags, params):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.point = (x,y)
			self.newpoint = True

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

	def __segmentNgetCorners__(self,img):

		# region segmentation
		lowerbound = (self.hue[0],self.sat[0],self.value[0])
		upperbound = (self.hue[1],self.sat[1],self.value[1])
		imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(imghsv,lowerbound,upperbound)
		kernel = np.ones((3,3),np.uint8)
		# mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
		mask = cv2.dilate(mask,None,iterations = 1)	
		cv2.imshow('segmentation',mask) 
		
		cnt, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(img, cnt, -1, (0,0,0))
		i = 0
		centroids = []
		for j in cnt:
			j = j[:,0,:]
			centroids.append((i, np.mean(j,axis=0)))
			i = i+1
		centroids = sorted(centroids, key = itemgetter(1),cmp = self.cmpPoints)
		if not centroids:
			return None,None,None


		imgheight = img.shape[0]
		cen = []
		for x in centroids:
			if x[1][1] > FRAC_IMG_ROT_HIGH*imgheight and x[1][1] < FRAC_IMG_ROT_LOW*imgheight:
				cen.append(x)
		if not cen:
			return None,None,None

		i = 1
		maxarea = cv2.contourArea(cnt[cen[0][0]])
		imax = 0
		while i < len(cen):
			area = cv2.contourArea(cnt[cen[i][0]])
			if area > maxarea:
				maxarea = area
				imax = i
			i = i + 1
		
		if self.areamax < maxarea:
			self.areamax = maxarea
		elif maxarea < FRAC_AREAMAX_ROT*self.areamax:
			print 'area too small. Discarded'
			return None,None,None
		region = cen[imax]
		self.area = maxarea
		# cv2.circle(img,(self.centroid[0],self.centroid[1]),4,(0,255,0))	
		
		# self.centroid = np.int32(region[1])
		# cv2.circle(img,(self.centroid[0],self.centroid[1]),4,(255,255,0))	
		


		approxPoly = cv2.approxPolyDP(cnt[region[0]],6,True)
		approxPoly = approxPoly[:,0,:]
		# if approxPoly != None:
		# 	for k in list(approxPoly):
		# 		cv2.circle(img,(k[0],k[1]),1,(0,255,0),-1)
		# 	cv2.polylines(img, [approxPoly], True, (0,0,255),1) 

		x,y,w,h = cv2.boundingRect(cnt[region[0]])
		roi = imghsv[y:y+h,x:x+w]

		hr,sr,vr = cv2.split(roi)
		mean_hue =  np.mean(hr)
		stddev_hue = np.std(hr)
		mean_value = np.mean(vr)
		std_value = np.std(vr)
		mean_sat = np.mean(sr)
		std_sat = np.std(sr)
		# self.hue[0] = mean_hue - 2*stddev_hue
		# self.hue[1] = mean_hue + 2*stddev_hue

		# self.sat[0] = mean_sat - 3*std_sat
		# self.sat[1] = mean_sat + 3*std_sat

		# self.value[0] = mean_value - 3*std_value
		# self.value[1] = mean_value + 3*std_value
		cv2.imshow('corners',img)
		cv2.waitKey(1)

		return approxPoly,self.area,self.centroid

	def cmpPointsToPrevious(self,p1,p2):
		if self.__distance__(p1,self.centroid) > self.__distance__(p2,self.centroid):
			return 1;
		elif self.__distance__(p1,self.centroid) < self.__distance__(p2,self.centroid):
			return -1;
		return 0;

	def cmpPoints(self,point1,point2):
		if point1[1] > point2[1]:
			return 1;
		elif point2[1] > point1[1]:
			return -1;
		elif point1[0] > point2[0]:
			return 1;
		elif point1[0] < point2[0]:
			return -1;
		return 0 

	def __distance__(self,p1,p2):
		p = p1-p2
		return np.linalg.norm(p)

	def __select_color__(self,event, x, y, flags, params):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.point = (x,y)
			self.newpoint = True

	def detect(self,newframe):
		hsv = cv2.cvtColor(newframe,cv2.COLOR_BGR2HSV)
		corners = self.__segmentNgetCorners__(hsv)

	#frameid should identify the order of the frames in the video
	def calibrate(self,frame,frameid):
		corners,area,centroid = self.__segmentNgetCorners__(frame)
		if corners != None:
			c = list(corners)
			c = sorted(corners, cmp = self.cmpPoints)
			r = self.__distance__(c[0],c[len(c)-1])
			print '%d:' % frameid
			print r
			self.frames.update({frameid:(r,area,centroid)})

	def getFrameInfo(self,frameid):
		return self.frames[frameid]


class stick:
	def __init__(self, id, frame1):
		self.id = id
		self.face1 = stickTracker(frame1,'stick %d'%id)
		self.rotations1 = dict()
		self.frames = []			#list of the frames used for calibrating.
		self.shape = frame1.shape

	def calibrate(self,frame,frameid):
		self.face1.calibrate(frame,frameid)
		self.frames.append(frameid)
		cv2.destroyWindow('segmentation')
		cv2.destroyWindow('corners')

	# classify the centroid in 3 regions.
	# 1 40% left of the image, 2 20% center part of the image, 3 40% right part of the image.
	def __classify__(self,centroid,frameid):
		if centroid[0] < FRAC_REG_1*self.shape[1]:
			print 'Frame %d: Region 1' % frameid
			return 1;
		elif centroid[0] < FRAC_REG_2*self.shape[1]:
			print 'Frame %d: Region 2' % frameid
			return 2;
		else:
			print 'Frame %d: Region 3' % frameid
			return 3;

	def __partitionFrames__(self, face):
		f1 = face.frames.keys()
		f1.sort()
		i = 0
		l1 = []
		ratios = [face.frames[x][0] for x in f1]
		maxr = max(ratios)
		rsorted = sorted(ratios)
		
		start = 0
		idx = []
		nummin = 1

		minant = ratios.index(rsorted[i])
		idx.append(minant)
		i = 1

		while i < len(ratios) and nummin < 4:
			x = rsorted[i]
			if np.sign(x - ratios[(ratios.index(x) + 1)]) != np.sign(ratios[minant - 1] - ratios[minant]):
				idx.append(ratios.index(x))
				nummin = nummin + 1
			i = i + 1
		idx.sort()
		idx.append(len(ratios)-1)
		idx.insert(0,0)

		while len(idx) > 0:
			i = idx.pop(0)
			j = idx.pop(0)
			l = []
			for k in range(i,j+1):
				l.append(f1[k])
			l1.append(l)


		return l1,f1

	def __computeRotationTable__(self,partitions,face):
		i = 0
 		k = 0
 		l1 = partitions
 		f1 = face.frames.keys()
		f1.sort()
 		if not l1:
 			print 'Empty partition list'
 			return {}
 		l = l1[k]
 		rotations = {}
 		sign = 1
 		rot = []
 		i = 0;
 		offset = 0
 		# Cosine of the angle of rotation in rot
 		for l in partitions:
 			ratios = [face.frames[x][1] for x in l]
			rmax = max(ratios)	 
			for r in ratios:
				rot.append(sign*r/rmax)
			sign = sign * -1	
		
		idx = rot.index(min(rot))
		k = rot.index(1)
		nFrames = f1[k]
		while k <= idx:
			if nFrames == f1[k]:
				rotations.update({nFrames: np.arccos(rot[k])})
			else:
				gap = f1[k]-nFrames+1
				if nFrames-1 >= 0 and f1[k] <= self.nFrames:
					prevrot = rotations[nFrames-1]
					rotgap = np.abs(prevrot - np.arccos(rot[k]))
					rotjump = rotgap/gap
					while nFrames < f1[k]:
						print nFrames
						print gap 
						print 'prevrot'
						print prevrot
						print rotjump
						rotations.update({nFrames: prevrot + rotjump})
						nFrames += 1
						prevrot += rotjump
					rotations.update({nFrames: np.arccos(rot[k])})
			k += 1
			nFrames += 1
		
		while k < len(rot) and rot[k] != 1:
			if nFrames == f1[k]:
				rotations.update({nFrames: np.pi + np.arccos(-rot[k])})
			else:
				gap = f1[k]-nFrames+1
				if nFrames-1 >= 0 and f1[k] <= self.nFrames:
					prevrot = rotations[nFrames-1]
					rotgap = np.abs(prevrot - np.pi - np.arccos(-rot[k]))
					rotjump = rotgap/gap
					while nFrames < f1[k]:
						print nFrames
						print 'prevrot'
						print prevrot
						print rotjump
						rotations.update({nFrames: prevrot + rotjump})
						nFrames += 1
						prevrot += rotjump
					rotations.update({nFrames: np.pi + np.arccos(-rot[k])})
			k += 1
			nFrames += 1

		if k < len(rot) and rot[k] == 1:
			rotations.update({f1[k]: np.pi + np.arccos(-rot[k])})
 		return rotations 

	def computeRotations(self,refFrameid):

		# break down the list of frames in a list of lists based on the change in size, 
		# so each sublist have a monotonous behaviour
		###############################################################################
		l1,f1 = self.__partitionFrames__(self.face1)
		print 'l1 :' + str(l1)
		##########################################################
 		
 		# compute the rotations table for each face
 		self.rotations1 = self.__computeRotationTable__(l1,self.face1)
 		
	# frames is a list of imgs to use for computing the rotations. Each frame is then identified by their position on the list
	def calibrateFull(self,frames):
		id = 0
		for i in frames:
			self.calibrate(i,id)
			id = id + 1
		self.nFrames = id
		self.computeRotations(0)

	def getRotation(self,frameid):		# return the rotation in radians
		if not bool(self.rotations1):
			print 'Hey you should compute first the rotations'
			return None
		if frameid in self.rotations1:
			return self.rotations1[frameid]
		else:
			return None
	def nValidFrames(self):
		return len(self.rotations1.keys())

# frames = []
# i = 0
# img = cv2.imread("testimages8/calPattern%.4d.jpg" % i)
# stick = stick(0,img)
# while i <= 236:
# 	img = cv2.imread("testimages8/calPattern%.4d.jpg" % i)
# 	img = cv2.GaussianBlur(img,(5,5),0)
# 	frames.append(img)
# 	i = i + 5

# stick.calibrateFull(frames)
# # os.mkdir('./RotationResults')
# for i in range(stick.nFrames):
# 	if stick.getRotation(i) != None:
# 		degrees = '%.2f degrees' %(np.degrees(stick.getRotation(i)))	
# 		((_,height),_) = cv2.getTextSize(degrees,cv2.FONT_HERSHEY_SIMPLEX,1,1)
# 		cv2.putText(frames[i],degrees, (0,height), cv2.FONT_HERSHEY_SIMPLEX, 1, 255,thickness=1)
# 		# cv2.imshow(frame[i])
# 		# cv2.waitKey(0)
# 		cv2.imwrite('./RotationResults/Result_Rotation_%.2f.jpg' % np.degrees(stick.getRotation(i)),frames[i])

