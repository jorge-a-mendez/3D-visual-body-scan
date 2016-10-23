"""
	This file is meant to contain a draft of the functions needed to be implemented 
	in order to compute the geometrical information. This should be implemented later
	on in a more intuitive class.
"""

from __future__ import division	#force / to be float division, // needed for int division
import cv2
from numpy import *
from numpy.linalg import *
import matplotlib as plot

# every matrix/vector like datatype should be a numpy array.

# return the ray (as point in the infinity) corresponding to imgpoint in the camera reference.
def getRay(imgpoint,kmatrix):	
	invk = inv(kmatrix)
	ray = dot(invk,imgpoint)
	return vstack([ray,0])

# return the angle (radians) between the rays of two image points.
def getAngleRays(imgpoint1, imgpoint2, kmatrix):	
	invk = inv(kmatrix)
	iomega =  dot(invk.transpose(),invk)
	costheta = dot(imgpoint1.transpose(),dot(iomega,imgpoint2))
	costheta = costheta/(sqrt(dot(imgpoint1.transpose(),dot(iomega,imgpoint1)))*sqrt(dot(imgpoint2.transpose(),dot(iomega,imgpoint2))))
	return arccos(costheta)

# return the normal vector (as euclidean vector) to the plane backprojecting from imgline (in homogeneous coords)
def getPlaneNormal(imgline, kmatrix):			
	return dot(kmatrix.transpose(),imgline)

# return the euclidean line (as a list that contains a point and a vector) that goes through the camera center and the ray referenced to the global reference system
def getLine(ray, distanceToPerson, distanceToPersonRef, rotation):	
	dif = distanceToPersonRef - distanceToPerson
	center = vstack([0,0,dif,1])
	t = vstack([-distanceToPerson*sin(rotation),0, (1-cos(rotation))*distanceToPerson,0])
	center = center + t
	point1 = ray + center
	point1 = point1[0:2]
	point2 = 2*ray + center
	point2 = point2[0:2]
	vector = point1 - point2
	line = [center[0:2], vector]
	return line

# return the line corresponding to the projection of the line to the plane.
# line is a list return by getLine function and plane is a list as [point in the plane, normal vector to plane]
def projectLineToPlane(line, plane):
	n = plane[1]/norm(plane[1])
	linev = cross(n,cross(line[1], n))		# vector in the plane corresponding to the direction of the line projection
	#t = dot(n.transpose(),plane[0]-line[0])/dot(n.transpose(),line[1])
	point = line[0] - dot(q-p,n)*n
	return [point, linev]

# return a plane as a list [point in the plane, normal vector to plane]
# points are in 3D euclidean coordinates
def getPlaneFromPoints(point1, point2, point3):
	vec1 = point1 - point2
	vec2 = point1 - point3
	n = cross(vec1, vec2)
	return [point1, n]

# return the point where line and plane intersect, if they do
# line is a list in the form returned by getLine
# plane is a list in the form returned by getPlaneFromPoints
def intersectLineWithPlane(line, plane):
	l0 = line[0]
	l = line[1]
	p0 = plane[0]
	n = plane[1]
	ln = dot(l,n)
	if not ln == 0:			#if not parallel
		d = dot(p0-l0,n)/ln
		return d*l + l0

# Use top and bottom points of person
# return the distance of the camera from the person. The points are in homogeneous coordinates
# This assumes that the image is metricly rectified.
def getDistanceToPerson_TopBottom(topimgpoint, bottomimgpoint, kmatrix, personHeight):
	top = topimgpoint/topimgpoint[2]
	top = top[0:1]
	bottom = bottomimgpoint/bottomimgpoint[2]
	bottom = bottom[0:1]
	size = norm(top - bottom)
	return personHeight*kmatrix[1,1]/size

# Use top point of person and plane that constraints head position
# headplane is the plane returned by getPlaneFromPoints using InfLine_Img heads
# headray is the ray correspoding to the head, disregarding rotation/translation
def getDistanceToPerson_TopPlane(headray,headplane):
	line = getLine(headray,0,0,0)	#disregard rotation and translation
	toppoint = intersectLineWithPlane(line, headplane)
	return toppoint[2]

# return the rotation between two frames. 
# vanishingPoints1 is a tuple with the two vanishing points in the first frame. Equivalently for the vanishingPoints2
def getRotation(vanishingPoints1, vanishingPoints2, kmatrix):
	zeros = zeros((1,3))
	v1 = getRay(vanishingPoints1[0],kmatrix)
	v1 = v1[0:2]
	v2 = getRay(vanishingPoints2[0],kmatrix)
	v2 = v2[0:2]
	a1 = hstack([zeros, v1[1]*v2.transpose(), -v1[2]*v2.transpose()])
	a2 = hstack([v1[2]*v2.transpose(), zeros, -v1[0]*v2.transpose()])
	A = vstack([a1,a2])
	v1 = getRay(vanishingPoints1[1],kmatrix)
	v1 = v1[0:2]
	v2 = getRay(vanishingPoints2[1],kmatrix)
	v2 = v2[0:2]
	a1 = hstack([zeros, v1[1]*v2.transpose(), -v1[2]*v2.transpose()])
	a2 = hstack([v1[2]*v2.transpose(), zeros, -v1[0]*v2.transpose()])
	A = vstack([A,a1,a2])

	r = lstsq(A,zeros((4,1)))
	R = vstack([r[0:2].transpose(), r[3:5].transpose(), r[6:8].transpose()])		# rotation matrix between frames
	
	return arctan2(-R[3,1],sqrt(R[3,2]**2 + R[3,3]**2)), R


# return the 2D line in 2D homogeneous coordinates.
# line is a tuple (point, vector)
def getHomLine(line):getHomLine(i[1])
	return vstack([line[1][1],-line[1][0], -line[1][1]*line[0][0] + line[1][0]*line[0][1]])

def intersect(line1, line2):
	p = cross(line1, line2)
	p = p/p[2]
	return p

def distance(point1, point2):
	return (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2

# lines is a list of tuples of the form (camera center, vector1, vector2)
# brute force algorithm.
def computeConvexHull(lines): 
	centroid = zeros((2,1))
	l = []
	for i in lines:
		l.append((getHomLine(i[1]),getHomLine(i[2])))			# convert to hom coords
		centroid = centroid + i[0]
	centroid = centroid / len(lines)							# compute centroid
	p = []

	# find best fit convex hull vertexes
	while len(l) != 0:
		k = l.pop(0)
		p1 = []
		p2 = []
		for j in range(2):
			for i in l:
				ret = intersect(i(0),k(j))
				if ret[2] != 0:
					p1.append(ret)
				ret = intersect(i(1),k(j))
				if ret[2] != 0:
					p2.append(ret)
			if len(p1) == 1:
				p.append(p1[0])
			elif len(p1) != 0:
				x = p1.pop(0)
				y = p1.pop(0)
				xd = distance(x,centroid)
				yd = distance(y,centroid)
				if yd < xd:
					z = x
					zd = xd
					x = y
					y = z
					xd = yd
					yd = zd
				while len(p1) != 0:
					z = p1.pop(0)
					zd = distance(z,centroid)
					if zd < xd:
						x = z
						xd = zd
					elif zd < yd:
						y = z
						yd = zd
				p.append(x)
				p.append(y)
			if len(p2) == 1:
				p.append(p2[0])
			elif len(p2) != 0:
				x = p2.pop(0)
				y = p2.pop(0)
				xd = distance(x,centroid)
				yd = distance(y,centroid)
				if yd < xd:
					z = x
					zd = xd
					x = y
					y = z
					xd = yd
					yd = zd
				while len(p2) != 0:
					z = p2.pop(0)
					zd = distance(z,centroid)
					if zd < xd:
						x = z
						xd = zd
					elif zd < yd:
						y = z
						yd = zd
				p.append(x)
				p.append(y)
	return p, cv2.convexHull(p)