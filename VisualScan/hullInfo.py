from __future__ import division	#force / to be float division, // needed for int division
import cv2
from numpy import *
from numpy.linalg import *
import matplotlib as plot
import display3D as disp
	
def getRay(imgpoint,kmatrix):
	invk = inv(kmatrix)
	ip = array([imgpoint[0],imgpoint[1],1])
	ip = reshape(ip,(3,1))
	ray = dot(invk,ip)
	return vstack([ray,0])

def getLine(ray, distanceToPerson, distanceToPersonRef, xToPerson, rotation,homography):	
	R = array([[cos(rotation), 0, sin(rotation)],[0, 1, 0],[-sin(rotation), 0, cos(rotation)],[0,0,0]])
	ray = dot(R.transpose(),ray)
	ray = vstack([ray,0])
	dif = distanceToPersonRef - distanceToPerson
	center = vstack([xToPerson,0,dif,1])
	t = vstack([distanceToPerson*sin(rotation),0, (1-cos(rotation))*distanceToPerson,0])
	center = center + t
	ray = dot(homography,ray)
	point1 = ray + center
	point1 = point1[0:3]/point1[3]
	point2 = 2*ray + center
	point2 = point2[0:3]/point2[3]
	vector = point1 - point2
	line = [center[0:3]/center[3], vector]
	return line

def projectLineToPlane(line, plane):
	n = plane[1]/norm(plane[1])
	# ---------------------------------- DEBUGGING ---------------------------------------
	# print 'Line inclination wrt floor plane'
	# print 90 - degrees(arccos(dot(line[1].transpose()/norm(line[1]),n)))
	# print 'Line inclination wrt plane XZ'
	# print 90 - degrees(arccos(dot(line[1].transpose()/norm(line[1]),array([0,1,0]))))
	# -------------------------------------------------------------------------------------
	l = line[1][:,0]
	linev = cross(n,cross(l, n))		# vector in the plane corresponding to the direction of the line projection
	linev = reshape(linev,(3,1))
	n = reshape(n,(3,1))
	point = line[0] + dot((-line[0]+reshape(plane[0],(3,1))).transpose(),n)*n
	point = array(((point[0,0]),(point[2,0])))
	linev = array(((linev[0,0]),(linev[2,0])))
	return [point, linev]

def getPlanarHomography(plane):
	k = plane[1]
	i = array((k[1],-k[0],0))
	i = i/norm(i)
	j = cross(i,k)
	i = hstack([i,0])
	j = hstack([j,0])
	k = hstack([k,0])
	last_col = hstack([0,0,0,1])
	homography = array((i,k,j,last_col))
	return homography.transpose()


def getHomLine(line):
	return array([line[1][1],-line[1][0], -line[1][1]*line[0][0] + line[1][0]*line[0][1]])

def intersect(line1, line2):
	p = cross(line1, line2)
	if p[2] == 0:
		return None
	p = p/p[2]
	return p[0:2]

def distance(point1, point2):
	return (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2

def computeConvexHull(lines): 
	centroid = zeros(2)
	l = []
	vertices = []
	centers = []
	for i in lines:
		l.append((getHomLine((i[0],i[1])),getHomLine((i[0],i[2]))))			# convert to hom coords
		centroid = centroid + i[0]
		centers.append(i[0])

		# to draw the lines
		xyz1 = [i[0][0],i[0][1]]
		xyz2 = [xyz1[0]+i[1][0],0,xyz1[1]+i[1][1]]
		if distance(xyz1,centroid) < distance(xyz2,centroid):
			xyz2 = [xyz1[0]-500*i[1][0], 0, xyz1[1]-500*i[1][1]]
		else:
			xyz2 = [xyz1[0]+500*i[1][0], 0, xyz1[1]+500*i[1][1]]
		
		xyz3 = [xyz1[0]+i[2][0], 0, xyz1[1]+i[2][1]]
		if distance(xyz1,centroid) < distance(xyz3,centroid):
			xyz3 = [xyz1[0]-500*i[2][0], 0, xyz1[1]-500*i[2][1]]
		else:
			xyz3 = [xyz1[0]+500*i[2][0], 0, xyz1[1]+500*i[2][1]]

		xyz1 = [xyz1[0],0,xyz1[1]]
		vertices.append((xyz1,xyz2))
		vertices.append((xyz1,xyz3))

	# fig,ax = disp.show_lines(vertices)		# plot lines use for the convex hull computation

	centroid = centroid / len(lines)
	center_hull = cv2.convexHull(float32(array(centers)))							# compute centroid
	p = []
	points = []
	for k in range(len(l)-1):
		x = intersect(l[k][0],l[k+1][0])
		y = intersect(l[k][1],l[k+1][1])
		if cv2.pointPolygonTest(center_hull,tuple(x),False) > 0 and cv2.pointPolygonTest(center_hull,tuple(y),False) > 0:
			p.append(x)
			p.append(y)
		else:

			# to show lines that causes points to be behind the cameras.
			vertices = []
			xyz1 = [lines[k][0][0],lines[k][0][1]]
			xyz2 = [xyz1[0]+lines[k][1][0],0,xyz1[1]+lines[k][1][1]]
			if distance(xyz1,centroid) < distance(xyz2,centroid):
				xyz2 = [xyz1[0]-500*lines[k][1][0], 0, xyz1[1]-500*lines[k][1][1]]
			else:
				xyz2 = [xyz1[0]+500*lines[k][1][0], 0, xyz1[1]+500*lines[k][1][1]]

			xyz3 = [xyz1[0]+lines[k][2][0], 0, xyz1[1]+lines[k][2][1]]
			if distance(xyz1,centroid) < distance(xyz3,centroid):
				xyz3 = [xyz1[0]-500*lines[k][2][0], 0, xyz1[1]-500*lines[k][2][1]]
			else:
				xyz3 = [xyz1[0]+500*lines[k][2][0], 0, xyz1[1]+500*lines[k][2][1]]

			xyz1 = [xyz1[0],0,xyz1[1]]
			vertices.append((xyz1,xyz2))
			vertices.append((xyz1,xyz3))

			xyz1 = [lines[k+1][0][0],lines[k+1][0][1]]
			xyz2 = [xyz1[0]+lines[k+1][1][0],0,xyz1[1]+lines[k+1][1][1]]
			if distance(xyz1,centroid) < distance(xyz2,centroid):
				xyz2 = [xyz1[0]-500*lines[k+1][1][0], 0, xyz1[1]-500*lines[k+1][1][1]]
			else:
				xyz2 = [xyz1[0]+500*lines[k+1][1][0], 0, xyz1[1]+500*lines[k+1][1][1]]

			xyz3 = [xyz1[0]+lines[k+1][2][0], 0, xyz1[1]+lines[k+1][2][1]]
			if distance(xyz1,centroid) < distance(xyz3,centroid):
				xyz3 = [xyz1[0]-500*lines[k+1][2][0], 0, xyz1[1]-500*lines[k+1][2][1]]
			else:
				xyz3 = [xyz1[0]+500*lines[k+1][2][0], 0, xyz1[1]+500*lines[k+1][2][1]]

			xyz1 = [xyz1[0],0,xyz1[1]]
			vertices.append((xyz1,xyz2))
			vertices.append((xyz1,xyz3))


		
	x = float32(array(p))
	# disp.addpoints(fig,ax,x,reshape(centroid,(1,2)))		# add the points found from the line's intersection, over the lines plot.
	return x, cv2.convexHull(x)[:,0,:], centroid,points

# point is an x,z pair
def getEuclideanPoint(point,plane):
	p = plane[0]
	n = plane[1]/norm(plane[1])
	y = ((p[2]-point[1])*n[2] + (p[0]-point[0])*n[0])/n[1] + p[1]
	return array((point[0],y,point[1]))

def get3DPoint(points,plane):
	l = []
	for i in list(points):
		x = getEuclideanPoint(i,plane)
		l.append(x)
	return array(l)

def intersectLineWithPlane(line,plane):
	l0 = line[0]
	l = line[1]

	p0 = reshape(plane[0],(3,1))
	n = plane[1]/norm(plane[1])
	n = reshape(n,(3,1))
	ln = dot(l.transpose(),n)
	if ln != 0:			#if not parallel
		d = dot((p0-l0).transpose(),n)/ln
		return (d*l + l0)[:,0]