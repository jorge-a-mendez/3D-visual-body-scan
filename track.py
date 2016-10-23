import cv2,cv
import numpy as np
from matplotlib import pyplot as plt
# from scipy.stats import itemfreq
import sys
from operator import itemgetter

#open a window to select the color of the face to track
#keep track of the change of rotation between frames.


class boxFaceTracker:
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
		mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
		mask = cv2.dilate(mask,None,iterations = 1)	
		cv2.imshow('segmented box',mask) 
		
		cnt, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(img, cnt, -1, (0,0,0))
		i = 0
		centroids = []
		for j in cnt:
			j = j[:,0,:]
			centroids.append((i, np.mean(j,axis=0)))
			i = i+1
		centroids = sorted(centroids, key = itemgetter(1),cmp = self.cmpPointsToPrevious)
		if not centroids:
			return None,None,None
		region = centroids[0]
		self.area = cv2.contourArea(cnt[region[0]])
		# if self.area < 50:
			# print 'area small'
			# return None,None,None 
		cv2.circle(img,(self.centroid[0],self.centroid[1]),4,(0,255,0))	
		# if np.abs(self.centroid[1] - region[1][1]) > 0.1*img.shape[0]:
			# print 'Too big leap for the centroid'
			
			# return None,None,None
		self.centroid = np.int32(region[1])
		cv2.circle(img,(self.centroid[0],self.centroid[1]),4,(255,255,0))	
		


		approxPoly = cv2.approxPolyDP(cnt[region[0]],5,True)
		approxPoly = approxPoly[:,0,:]
		if approxPoly != None:
			if approxPoly.shape[0] != 4:
				# print 'Not a quadrilateral'
				return None,None,None
			for k in list(approxPoly):
				cv2.circle(img,(k[0],k[1]),1,(0,255,0),-1)
			cv2.polylines(img, [approxPoly], True, (0,0,255),1) 
		# else:
			# print 'Polygon not found' 

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
		print 'FOUND :D'
		cv2.imshow('corners',img)
		# cv2.waitKey(50)

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
		# hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		corners,area,centroid = self.__segmentNgetCorners__(frame)
		if corners != None:
			corners = cv2.convexHull(corners)
			c = list(corners)
			# print c
			h = self.__distance__(c[0],c[1]) + self.__distance__(c[2],c[3])
			v = self.__distance__(c[1],c[2]) + self.__distance__(c[3],c[0])
			r = h/v
			# if self.rmax[1] < r:
			# 	self.rmax = (frameid,r)
			# 	print self.rmax
			self.frames.update({frameid:(r,area,centroid)})

	def getFrameInfo(self,frameid):
		return self.frames[frameid]


class box:
	def __init__(self, boxid, frame1, frame2):
		self.id = boxid
		self.face1 = boxFaceTracker(frame1,'face1 box %d'%boxid)
		self.face2 = boxFaceTracker(frame2,'face2 box %d'%boxid)
		self.rotations1 = dict()
		self.rotations2 = {}
		self.frames = []			#list of the frames used for calibrating.
		self.shape = frame1.shape

	def calibrate(self,frame,frameid):
		self.face1.calibrate(frame,frameid)
		self.face2.calibrate(frame,frameid)
		self.frames.append(frameid)


	# classify the centroid in 3 regions.
	# 1 40% left of the image, 2 20% center part of the image, 3 40% right part of the image.
	def __classify__(self,centroid,frameid):
		if centroid[0] < 0.4*self.shape[1]:
			print 'Frame %d: Region 1' % frameid
			return 1;
		elif centroid[0] < 0.6*self.shape[1]:
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
		signl1 = []
		if len(f1) > 1:
			subl1 = [f1[i]]
			area1p = face.getFrameInfo(f1[i])[1]
			regp = self.__classify__(face.getFrameInfo(f1[i])[2],i)
			i = 1
			area1c = face.getFrameInfo(f1[i])[1]
			regc = self.__classify__(face.getFrameInfo(f1[i])[2],i)
			sign1 = np.sign(area1p-area1c)
			if sign1 == np.sign(area1p-area1c) or regp == regc:
				subl1.append(f1[i])
			else:
				signl1.append(sign1)
				l1.append(subl1)
				subl1 = [f1[i]]
				sign1 = np.sign(area1p-area1c)
			i = i + 1
			area1p = area1c
			regp = regc
			while i < len(f1):
				area1c = face.getFrameInfo(f1[i])[1]
				regc = self.__classify__(face.getFrameInfo(f1[i])[2],i)
				if sign1 == np.sign(area1p-area1c) or regp == regc:
					subl1.append(f1[i])
				else:
					signl1.append(sign1)
					l1.append(subl1)
					subl1 = [f1[i]]
					sign1 = np.sign(area1p-area1c)
				i = i + 1
				area1p = area1c
				regp = regc
			l1.append(subl1)
		return l1,f1

	def __computeRotationTable__(self,partitions,face):
		i = 0
 		k = 0
 		l1 = partitions
 		if not l1:
 			print 'Empty partition list'
 			return {}
 		l = l1[k]
 		rotations = {}
 		ratios = [face.frames[x][1] for x in l]
		rmax = max(ratios)
		idx = ratios.index(rmax)
 		fmax = l[idx]
		offset = 0
		j = 0
		for i in l:
			if j < idx:
				rotations.update({i:offset-np.degrees(np.arccos(ratios[j]/rmax))})
			else:
				rotations.update({i:offset+np.degrees(np.arccos(ratios[j]/rmax))})
			j = j + 1
 		fmaxant = fmax
 		rmaxant = rmax
 		k = k + 1
 		while k < len(l1):
 			l = l1[k]
 			ratios = [face.frames[x][1] for x in l]
 			rmax = max(ratios)
 			idx = ratios.index(rmax)
 			fmax = l[idx]

 			changex = np.abs(self.__classify__(face.frames[fmax][2],fmax) - self.__classify__(face.frames[fmaxant][2],fmaxant))
 			if changex == 2:
 				offset = offset + 180
 			elif changex == 1:
 				offset = offset + 90

 			j = 0
 			for i in l:
 				if j < idx:
 					rotations.update({i:offset-np.degrees(np.arccos(ratios[j]/rmax))})
 				else:
 					rotations.update({i:offset+np.degrees(np.arccos(ratios[j]/rmax))})
 				j = j + 1
 			fmaxant = fmax
 			rmaxant = rmax
 			i = i + 1
 			k = k + 1

 		return rotations 

	def computeRotations(self,refFrameid):

		# break down the list of frames in a list of lists based on the change in size, 
		# so each sublist have a monotonous behaviour
		###############################################################################
		l1,f1 = self.__partitionFrames__(self.face1)
		l2,f2 = self.__partitionFrames__(self.face2)
		print 'l1 :' + str(l1)
		print 'l2 :' + str(l2)
		##########################################################

		# At this point for each sublist we can have the maximum ratio so we know the change of orientation for the
 		# visible face of the box
 		
 		# compute the rotations table for each face
 		self.rotations1 = self.__computeRotationTable__(l1,self.face1)
 		self.rotations2 = self.__computeRotationTable__(l2,self.face2)
 		
		f1 = set(f1)
		f2 = set(f2)
		mincommonframe = min(f1&f2)
		offset = self.rotations1[mincommonframe] 
		for key in self.rotations1:
			self.rotations1[key] -= offset
		offset = self.rotations2[mincommonframe]
		for key in self.rotations2:
			self.rotations2[key] -= offset
		print 'Frame %d set as reference' % mincommonframe

		return mincommonframe

	# frames is a list of imgs to use for computing the rotations. Each frame is then identified by their position on the list
	def calibrateFull(self,frames):
		id = 0
		for i in frames:
			self.calibrate(i,id)
			id = id + 1
		self.computeRotations(0)

	def getRotation(self,frameid):		# return the rotation in radians
		if not bool(self.rotations1) and not bool(self.rotations2):
			print 'Hey you should compute first the rotations'
			return None
		x = self.rotations1.get(frameid)
		if x != None:
			y = self.rotations2.get(frameid)
			if y != None:
				return (x+y)/2
			else: 
				return x 
		else:
			y = self.rotations2.get(frameid)
			if y != None:
				return y
		print 'Couldn\'t find the frame %d!' % frameid
		return float('NaN')



frames = []
i = 36
img = cv2.imread("testimages6/calPattern%.4d.jpg" % i)
# img = cv2.GaussianBlur(img,(5,5),1) 
img2 = cv2.imread("testimages6/calPattern%.4d.jpg" % 96)
# img2 = cv2.GaussianBlur(img2,(5,5),2) 
box1 = box(1,img,img)
while i <= 250:
	img = cv2.imread("testimages6/calPattern%.4d.jpg" % i)
	img = cv2.GaussianBlur(img,(5,5),0)
	frames.append(i)
	box1.calibrate(img,i)
	i = i + 5
mincommonframe = box1.computeRotations(0)
# print 'rotations1'
# print box1.rotations1
# print 'rotations2'
# print box1.rotations2
# print mincommonframe

for i in frames:
	print 'Frame %d: %f degrees' %(i, box1.getRotation(i))	