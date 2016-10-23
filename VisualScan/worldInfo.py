''' ================================================================================================
		This module contains the methods to compute all the information related to the 
		real world coordinates.
    ================================================================================================ '''


from __future__ import division	#force / to be float division, // needed for int division
import cv2
from numpy import *
from numpy.linalg import *
import matplotlib as plot

#---------GLOBAL VARIABLES FOR CLASS-CALLBACKS INTERFACING----------#
goto_overlap = False
goto_done = False
HEAD = 0
FEET = 1
person_numbers = ['Center','Left','Right']
head_or_feet = ['HEAD', 'FEET']

drawing = False
ix,iy = -1,-1
FoundFoot = False
point = (0,0)
#--------------------------------------------------------------------#

# Public methods:
#	select_heads_feet: allows user to select the areas of heads and feet
#	get_DistanceToPerson: receives the image point corresponding to
#						a head and returns the person-to-camera distance
class worldInfo:
	def __init__(self, imgs,height = 0, kmtrx = 0):
		self.imgs = imgs
		self.person_height = height
		self.kmtrx = kmtrx

	def select_heads_feet(self):
		# global goto_overlap, goto_done
		# self.window = 'Heads&Feet'
		# cv2.namedWindow(self.window)
		# cv2.createTrackbar('minVal',self.window, 0, 500, goto_overlapImgs)
		# cv2.createTrackbar('maxVal',self.window,500,500,goto_overlapImgs)
		# cv2.createTrackbar('DONE',self.window,0,1,goto_doneThresholding)
		# self.__overlapImgs__()
		# self.ix, self.iy = -1,-1
		# self.destroyWindow = False
		# self.drawing = False
		# self.goto_select = False
		# self.done_finding = False
		# self.heads_feet = [[0 for i in xrange(3)] for i in xrange(2)]
		# self.heads3D = [0 for i in xrange(3)]
		# self.head_feet = HEAD
		# self.person_number = 0
		# while(1):
		# 	cv2.waitKey(1)
		# 	if self.destroyWindow == True:
		# 		self.__new_window__()
		# 		self.destroyWindow = False
		# 	if goto_overlap == True:
		# 		self.__overlapImgs__()
		# 		goto_overlap = False
		# 	if goto_done == True:
		# 		self.__doneThresholding__()
		# 		# cv2.imwrite("Results_OverlappedImage.jpg",self.overlapped)
		# 		goto_done = False
		# 		# cv2.destroyAllWindows()
		# 		# break
		# 	if self.done_finding == True:
		# 		cv2.imshow(self.window, self.overlapped)
		# 		cv2.waitKey(2000)
		# 		cv2.destroyWindow(self.window)
		# 		self.__find3Dpoints__()
		# 		self.__findHeadsPlane__()

		# 		a = array(self.heads_feet[HEAD])
		# 		savetxt("heads.txt",a)

		# 		a = array(self.heads_feet[FEET])
		# 		savetxt("feet.txt",a)
		# 		break

		# LOAD FROM FILES FOR DEBUGGING------------------------------

		self.heads_feet = [[0 for i in xrange(3)] for i in xrange(2)]
		self.heads3D = [0 for i in xrange(3)]
		a = loadtxt("heads.txt")
		a = list(a)
		self.heads_feet[HEAD] = a
		a = loadtxt("feet.txt")
		a = list(a)
		self.heads_feet[FEET] = a 

		self.__find3Dpoints__()
		self.__findHeadsPlane__()



	def __findNormalToFloor__(self, kmatrix):
		headcenter = append(self.heads_feet[HEAD][0],1)
		feetcenter = append(self.heads_feet[FEET][0],1)
		headleft = append(self.heads_feet[HEAD][1],1)
		headright = append(self.heads_feet[HEAD][2],1)
		feetleft = append(self.heads_feet[FEET][1],1)
		feetright = append(self.heads_feet[FEET][2],1)

		lineHead1 = cross(headcenter,headleft)
		lineHead2 = cross(headcenter,headright)
		lineFeet1 = cross(feetcenter,feetleft)
		lineFeet2 = cross(feetcenter,feetright)
		vpoint1 = cross(lineHead1,lineFeet1)
		vpoint2 = cross(lineHead2,lineFeet2)
		vline = cross(vpoint2,vpoint1)

		n = dot(kmatrix.transpose(),vline)
		n = n/norm(n)
		return reshape(n,(3,1))


	def __overlapImgs__(self):
		person = []
		self.minVal = cv2.getTrackbarPos('minVal',self.window)
		self.maxVal = cv2.getTrackbarPos('maxVal',self.window)
		for f in self.imgs:
			temperson = cv2.Canny(f,self.minVal,self.maxVal)
			person.append(temperson)
		self.overlapped = person[0] | person[1] | person[2]
		cv2.imshow(self.window,self.overlapped)

	def __doneThresholding__(self):
		self.__overlapImgs__()
		self.goto_select = True
		self.destroyWindow = True

	def __new_window__(self):
		global head_or_feet, person_numbers
		img = self.overlapped.copy()
		cv2.destroyWindow(self.window)
		cv2.namedWindow(self.window)
		cv2.setMouseCallback(self.window,select_area,(self))
		text = head_or_feet[self.head_feet] + ' ' + person_numbers[self.person_number]
		((_,height),_) = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,1)
		cv2.putText(img,text,(0,height),cv2.FONT_HERSHEY_SIMPLEX,1,255,thickness=1)
		cv2.imshow(self.window, img)

	def __find3Dpoints__(self):
		ux = self.kmtrx[0,2]
		uy = self.kmtrx[1,2]
		f = self.kmtrx[1,1]
		for i in range(3):
			img_height = abs(self.heads_feet[HEAD][i][1]-self.heads_feet[FEET][i][1])
			img_x = self.heads_feet[HEAD][i][0] - ux
			img_y = self.heads_feet[HEAD][i][1] - uy
			# x = -self.person_height*img_x/img_height
			# y = -self.person_height*img_y/img_height
			pointimg = (self.heads_feet[HEAD][i][0],self.heads_feet[HEAD][i][1])
			z = self.person_height*f/img_height
			pointline = self.__ImgPoint2RayLine__(pointimg)
			point3D = self.__intersectLineWithXYPlane__(pointline,z)
			point3D = point3D[:,0]
			self.heads3D[i] = array(point3D)

	def __findHeadsPlane__(self):
		vec1 = self.heads3D[0] - self.heads3D[1]
		vec2 = self.heads3D[0] - self.heads3D[2]
		n = cross(vec1, vec2)
		n = n/norm(n)
		# n = array((0,1,0))	# ignore inclination of plane	
		self.heads_plane = [self.heads3D[0],n]

		n2 = self.__findNormalToFloor__(self.kmtrx)
		print 'Dot product of normal vectors'
		print degrees(arccos(dot(n2.transpose(),reshape(n,(3,1)))))

	def saveResults(self,headimg_frame0,frame180,headimg_frame180):
		global point

		n = reshape(self.heads_plane[1],(3,1))
		n2 = vstack((0,1,0))
		inclination = degrees(arccos(dot(n2.transpose(),reshape(n,(3,1)))))

		self.__findFeetOnce__(frame180.copy())

		headimg_0 = (self.heads_feet[HEAD][0][0],self.heads_feet[HEAD][0][1])
		feetimg_0 = (self.heads_feet[FEET][0][0],self.heads_feet[FEET][0][1])
		height_0 = self.get_head2PointDistance(headimg_0,feetimg_0) 
		distance_0 = self.get_DistanceToPerson(headimg_0)
		refeet_0 = self.get_imgHeightAtHeadDistance(headimg_0,self.person_height)
		
		headimg_1 = (self.heads_feet[HEAD][1][0],self.heads_feet[HEAD][1][1])
		feetimg_1 = (self.heads_feet[FEET][1][0],self.heads_feet[FEET][1][1])
		height_1 = self.get_head2PointDistance(headimg_1,feetimg_1) 
		distance_1 = self.get_DistanceToPerson(headimg_1)
		refeet_1 = self.get_imgHeightAtHeadDistance(headimg_1,self.person_height)
	
		headimg_2 = (self.heads_feet[HEAD][2][0],self.heads_feet[HEAD][2][1])
		feetimg_2 = (self.heads_feet[FEET][2][0],self.heads_feet[FEET][2][1])
		height_2 = self.get_head2PointDistance(headimg_2,feetimg_2) 
		distance_2 = self.get_DistanceToPerson(headimg_2)
		refeet_2 = self.get_imgHeightAtHeadDistance(headimg_2,self.person_height)

		height_3 = self.get_head2PointDistance(headimg_frame180,point)
		distance_3 = self.get_DistanceToPerson(headimg_frame0)
		refeet_3 = self.get_imgHeightAtHeadDistance(headimg_frame180,self.person_height)

		str_inclination = 'Inclination of the heads plane: %.3f\n' % inclination
		str_h_0 = 'Center height: %.2f\n' % height_0
		str_h_1 = 'Left height: %.2f\n' % height_1
		str_h_2 = 'Right height: %.2f\n' % height_2
		str_h_3 = '180deg height: %.2f\n\n\n' % height_3
		str_d_0 = 'Center distance: %.2f\n' % distance_0
		str_d_1 = 'Left distance: %.2f\n' % distance_1
		str_d_2 = 'Right distance: %.2f\n' % distance_2 	 
		str_d_3 = '0deg distance: %.2f\n\n\n' % distance_3
		str_r_0 = 'Center original, recomputed: %d, %d\n' % (feetimg_0[1],refeet_0)
		str_r_1 = 'Left original, recomputed: %d, %d\n' % (feetimg_1[1],refeet_1)
		str_r_2 = 'Right original, recomputed: %d, %d\n' %(feetimg_2[1],refeet_2)
		str_r_3 = '180deg original, recomputed: %d, %d\n\n\n' %(point[1],refeet_3)


		f = open('Results_worldInfo.txt','w')

		f.write('Test reference values computation\n')
		f.write(str_inclination)
		f.write(self.heads_plane.__str__())
		f.write('\n\n\n')

		f.write('Test function Head2PointDistance\n')
		f.write(str_h_0)
		f.write(str_h_1)
		f.write(str_h_2)
		f.write(str_h_3)

		f.write('Test function DistanceToPerson\n')
		f.write(str_d_0)
		f.write(str_d_1)
		f.write(str_d_2)
		f.write(str_d_3)

		f.write('Test function HeightAtHeadDist\n')
		f.write(str_r_0)
		f.write(str_r_1)
		f.write(str_r_2)
		f.write(str_r_3)

	def __findFeetOnce__(self, frame):
		cv2.namedWindow('SelectFeet')
		edges = cv2.Canny(frame,self.minVal,self.maxVal)
		cv2.setMouseCallback('SelectFeet',select_area_once,(self,edges))
		cv2.imshow('SelectFeet', edges)

		while(1):
			cv2.waitKey(1)
			if FoundFoot == True:
				cv2.waitKey(500)
				cv2.destroyWindow('SelectFeet')
				break		

	def __intersectLineWithHeadsPlane__(self,line):
		l0 = line[0]
		l = line[1]

		p0 = reshape(self.heads_plane[0],(3,1))
		n = self.heads_plane[1]
		n = reshape(n,(3,1))
		ln = dot(l.transpose(),n)
		if ln != 0:			#if not parallel
			d = dot((p0-l0).transpose(),n)/ln
			return d*l + l0

	def __intersectLineWithXYPlane__(self,line,z):
		l0 = line[0]
		l = line[1]
		p0 = reshape([0,0,z],(3,1))
		n = reshape([0,0,1],(3,1))
		ln = dot(l.transpose(),n)
		if ln != 0:
			d = dot((p0-l0).transpose(),n)/ln
			return d*l + l0


	def select_next(self):
		if self.person_number == 2:
			self.head_feet = self.head_feet + 1
		self.person_number = (self.person_number + 1) % 3
		if self.head_feet > 1:
			self.done_finding = True
			self.goto_select = False
		else:
			self.destroyWindow = True

	def find_point(self,(x,y)):
		if self.head_feet == HEAD:
			i = self.ix
			j = self.iy
			while j < y:
				while i < x:
					if self.overlapped[j,i] != 0:
						return (i,j)
					i = i + 1
				j = j + 1
				i = self.ix
		elif self.head_feet == FEET:
			i = x
			j = y
			while j > self.iy:
				while i > self.ix:
					if self.overlapped[j,i] != 0:
						return (i,j)
					i = i - 1
				j = j - 1
				i = x

	def change_ixiy(self,x,y):
		self.ix, self.iy = x,y

	def save_point(self,point):
		self.heads_feet[self.head_feet][self.person_number] = point

	# Distance between the person and the camera center.
	def get_DistanceToPerson(self,headimg):
		line = self.__ImgPoint2RayLine__(headimg)
		head3D = self.__intersectLineWithHeadsPlane__(line)
		print 'DistanceToPerson'
		print head3D[2]
		return head3D[2]

	# Horizontal deviation of the person to the camera center.
	def get_XToPerson(self,headimg):
		line = self.__ImgPoint2RayLine__(headimg)
		head3D = self.__intersectLineWithHeadsPlane__(line)
		return head3D[0]

	# Real distance between two image points.
	def get_head2PointDistance(self,headimg,pointimg):
		headline = self.__ImgPoint2RayLine__(headimg)
		head3D = self.__intersectLineWithHeadsPlane__(headline)
		z = head3D[2]
		pointline = self.__ImgPoint2RayLine__(pointimg)
		point3D = self.__intersectLineWithXYPlane__(pointline,z)
		return -head3D[1] + point3D[1]

	# Returns the image height corresponding to real distance from the head.
	def get_imgHeightAtHeadDistance(self,headimg,distance):
		headline = self.__ImgPoint2RayLine__(headimg)
		head3D = self.__intersectLineWithHeadsPlane__(headline)
		point3D = array([[head3D[0,0]], [head3D[1,0]+distance], [head3D[2,0]]])
		zero = zeros((3,1))
		KI0 = self.kmtrx
		pointImg = dot(KI0,point3D)
		pointImg = pointImg//pointImg[2,0]
		return pointImg[1,0]


	# Method that returns the ray for the top of the head.
	def __ImgPoint2RayLine__(self,headimg):

		h = vstack([headimg[0],headimg[1],1])
		headray = vstack([dot(inv(self.kmtrx),h),0])
		point1 = headray
		point1 = point1[0:3]
		
		point2 = 2*headray
		point2 = point2[0:3]
		vector = point2 - point1
		line = [zeros((3,1)), vector]
		return line	# from getLine, making center=000

#------------------CALLBACKS FOR THE CLASS WORLDINFO GUI-------------------#
def select_area(event, x, y, flags, (world)):
	temp_img = world.overlapped.copy()
	rect = ones_like(temp_img)
	if world.goto_select == True:
		if event == cv2.EVENT_LBUTTONDOWN:
			world.drawing = True
			# ix,iy = x,y
			world.change_ixiy(x,y)

		elif event == cv2.EVENT_MOUSEMOVE:
			if world.drawing == True:
				cv2.rectangle(rect, (world.ix,world.iy),(x,y),255,-1)
				temp_img = cv2.addWeighted(temp_img, 1, rect, .3,0)
				cv2.imshow(world.window, temp_img)

		elif event == cv2.EVENT_LBUTTONUP:
			if world.drawing == True:
				world.drawing = False
				# cv2.rectangle(temp_img, (world.ix,world.iy),(x,y),0,0)
				# temp_img = cv2.addWeighted(temp_img, 1, rect, .3,0)
				
				point = world.find_point((x,y))
				cv2.circle(world.overlapped, point,5,255,-1)
				cv2.imshow(world.window, temp_img)
				world.save_point(point)
				world.select_next()


def goto_overlapImgs(x):
	global goto_overlap
	goto_overlap = True

def goto_doneThresholding(done):
	global goto_done
	if done == 1:
		goto_done = True

def select_area_once(event, x, y, flags, (world,frame)):
	global drawing, FoundFoot, ix, iy, point
	rect = ones_like(frame)
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix,iy = x,y
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			cv2.rectangle(rect, (ix,iy),(x,y),255,-1)
			temp_img = cv2.addWeighted(frame, 1, rect, .3,0)
			cv2.imshow('SelectFeet', frame)
	elif event == cv2.EVENT_LBUTTONUP:
		if drawing == True:
			drawing = False
			# edges = cv2.Canny(frame,world.minVal,world.maxVal)
			point = find_feet_once((ix,iy),(x,y),frame)
			cv2.circle(frame, point,5,255,-1)
			cv2.imshow('SelectFeet', frame)
			FoundFoot = True

def find_feet_once((ix,iy),(x,y),frame):
	i = x
	j = y
	while j > iy:
		while i > ix:
			if frame[j,i] != 0:
				return (i,j)
			i = i - 1
		j = j - 1
		i = x	

#---------------------------------------------------------------------#

# img = []
# i = 0
# while(i < 3):

# 	title = "1InfLine_%d.jpg" %(i)
# 	gray = cv2.imread(title, 0)
# 	gray = cv2.resize(gray, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
# 	img.append(gray)

# 	i = i + 1

# k = loadtxt("OnePlus_cal.txt")
# world = worldInfo(img,180,k)
# world.select_heads_feet()
# print world.heads_feet
# print world.heads3D
# print world.heads_plane