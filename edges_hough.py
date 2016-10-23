import cv2,cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import itemfreq
import sys
#np.set_printoptions(threshold=np.nan)

def indexes(matrix, value):
	index = np.nonzero(matrix == value);
	indexes = np.vstack([index[0],index[1]])
	indexes = indexes.transpose()
	#indexes.sort()
	# print indexes 
	return indexes

class regions:
	def __init__(self, markers, region):
		self.mask = np.zeros(markers.shape,np.uint8)
		self.region = region
		points = indexes(markers,region)
		self.centroid = np.sum(points,axis = 0)/points[:,0].size
		self.area = points[:,0].size
		self.mask[markers == region] = 255
		self.contours, level = cv2.findContours(self.mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
	def getMask(self):
		return self.mask
	def getArea(self):
		return self.area
	def getCentroid(self):
		return self.centroid
	def getContour(self):
		# print self.contours
		return self.contours
	def __str__(self):
		return "Region {0}: Area {1}, centroid {array}".format(self.region,self.area,array=self.centroid)
	def __repr__(self):
		return self.__str__()

def getCentroid(reg):
	# print reg.getCentroid()
	return reg.getCentroid()
def cmpPoints(point1,point2):
	if point1[0] > point2[0]:
		return 1;
	elif point2[0] > point1[0]:
		return -1;
	elif point1[1] > point2[1]:
		return 1;
	elif point1[1] < point2[1]:
		return -1;
	return 0
def getArea(reg):
	return reg.getArea()


def getBoxCorners(regions, corners,img):
	regionl = sorted(regions,cmp = cmpPoints,key=getCentroid)			# sort the regions top to down 
	valid = []
	l = []
	# filter the regions by size and position in the image.
	for i in regionl:
		if i.getCentroid()[0] > 10 and i.getCentroid()[0] < 100:
			if i.getArea() > 20 and i.getArea() < 300:
				valid.append(i)


	# if len(valid) > 2:
	# 	x = valid[0]
	# 	y = valid[1]
	# 	sim = cv2.matchShapes(x.getContour()[0],y.getContour()[0],2,0)
	# 	for i in range(len(valid)):
	# 		for j in range(i+1,len(valid)):
	# 			if j < len(valid):
	# 				sim1 = cv2.matchShapes(valid[i].getContour()[0], valid[j].getContour()[0],2,0)
	# 				if sim1 < sim:
	# 					x = valid[i]
	# 					y = valid[j]
	# 					sim = sim1
	# 	valid = [x,y]


	print valid
	for i in valid:
		print i
		mask = i.getMask()
		# mask = cv2.dilate(mask,None)
		# mask[mask == 255] = f[mask == 255]
		# cv2.imshow('region mask', mask)
		# cv2.waitKey(0)

		cornersReg = corners.copy()
		contours = i.getContour()
		# mask = cv2.Canny(mask,100,50)
		cornersReg[mask == 0] = 0
		for j in contours:
			approxPoly = cv2.approxPolyDP(j,3.3,True)
			approxPoly = approxPoly[:,0,:]
			for k in list(approxPoly):
				cv2.circle(img,(k[0],k[1]),1,(0,0,255),-1)
		cv2.imshow('mask',img)
		cv2.waitKey(0)
		# for i in range(4):
		# 	print "Corner %d" % i
		# 	print np.nonzero(cornersReg == cornersReg.max())
		# 	t = np.nonzero(cornersReg == cornersReg.max())
		# 	point = np.hstack([t[0],t[1]])
		# 	cornersReg[cornersReg == cornersReg.max()] = 0
		# 	l.append(point)
	return l




k = 117;
while k <= 350:
	filename = "testimages5/calPattern%.4d.jpg" % k;
	img = cv2.imread(filename);
	img = cv2.GaussianBlur(img,(5,5),0) 
	f = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
	f = cv2.equalizeHist(f)
	b,g,r = cv2.split(img)
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	edges = cv2.Canny(f,100,50)

	# ====================== Color based segmentation =====================
	# lowerbound = (130,0,0)
	# upperbound = (179,255,255)
	# threshold = cv2.inRange(hsv,lowerbound,upperbound)
	# cv2.imshow('color',threshold)



	# ================= watershed segmentation =============================

	ret,bin = cv2.threshold(f,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	# noise removal 
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(bin,cv2.MORPH_OPEN,kernel,iterations=3)

	#sure bg

	surebg = cv2.dilate(opening,kernel,iterations = 1)

	# sure foreground
	distTransform = cv2.distanceTransform(opening,cv.CV_DIST_L2,0)
	ret,surefg = cv2.threshold(distTransform,0*distTransform.max(),255,0)

	#
	surefg = np.uint8(surefg)
	unknown = cv2.subtract(surebg,surefg)

	# marker labelling
	contours, level = cv2.findContours(surefg,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
	markers = np.zeros(surefg.shape,np.int32)
	i = 0
	compCount = 0
	while (i >=0):
		cv2.drawContours(markers,contours,i,compCount+1,-1,8,level)
		i = level[0,i,0]
		compCount = compCount + 1
	# markers = markers+1
	# markers[unknown == 255] = 0
	cv2.watershed(img,markers)
	img[markers==-1] = (255,0,0)
	# ======================================================================

	# ==================== blob detection ==================================

	# detector = cv2.SimpleBlobDetector()
	# keypoints = detector.detect(img)
	# im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# # cv2.imshow('blobs', im_with_keypoints)
	
	k = k + 1;

	f[edges == 255] = 255; 
	dst = cv2.cornerHarris(np.float32(markers),2,3,0.1);
	dst1 = cv2.cornerHarris(b,2,3,0.1);
	# dst = cv2.dilate(dst,None);
	# img[dst > 0.08*dst.max()] = (0,0,255);
	# img[dst1 > 0.005*dst1.max()] = (0,255,0);

	region = list(np.unique(markers))
	region.pop(0)
	markersc = np.zeros(markers.shape,np.uint8)
	markersc = cv2.cvtColor(markersc, cv2.COLOR_GRAY2BGR)
	regionl = []
	for reg in region:
		x = regions(markers,reg)
		regionl.append(x)
		# Show region masks
		# cv2.imshow('masks', x.getMask())
		# cv2.waitKey(0)

	# print regionl
	# corners = getBoxCorners(regionl,dst,img)
	# for j in corners:
	# 	# i = j.getCentroid()
	# 	img[j[0],j[1]] = (0,0,255) 
	# 	cv2.circle(img,(j[1],j[0]),3,(0,255,0))

	# img[corners > 0.1*corners.max()] = (0,255,0)	

	cv2.imshow('edges',edges)
	# cv2.imshow('thresh',bin)
	# cv2.imshow('opening',opening)
	# cv2.imshow('bg',surebg)
	# cv2.imshow('fg',np.uint8(surefg))
	# cv2.imshow('dt',distTransform)
	# cv2.imshow('watershed', markersc);
	cv2.imshow('segmentation',img)
	# cv2.imshow('bw',f)
	# cv2.imshow('red',r)
	# cv2.imshow('green',g)
	# cv2.imshow('blue',b)
	cv2.waitKey(1);
	# cv2.destroyAllWindows()