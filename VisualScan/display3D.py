from __future__ import division
import numpy as np
import cv2
import detection as dt
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(threshold = np.nan)
# points: list of 3d points
# [[x,y,z],[x,y,z],...,[x,y,z]]
def show_pointcloud(points1,points2 = None):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	points1 = np.array(points1)
	x = points1[:,0]
	z = points1[:,2]
	y = points1[:,1]

	ax.scatter(x,y,z)

	if points2 != None:
		x2 = points2[:,0]
		z2 = points2[:,2]
		y2 = points2[:,1]

		ax.scatter(x2,y2,z2,c='r')


	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()

def get2Darrays(points):
	points = np.array(points)
	# print points
	# points.sort(axis=1) #sort along the third (y) axis
	# print points

	size = points.shape[0]
	print size
	icount = 0
	fcount = 0
	# x = np.array([])
	# y = np.array([])
	# z = np.array([])
	x = []
	y = []
	z = []
	i = 0

	# print points
	while fcount < size-1:
		ix = points[icount,1]
		# print icount, ix
		fx = ix.copy()
		while fx == ix and fcount < size-1:
			fcount = fcount + 1
			fx = points[fcount,1]
		
		if fcount == size-1:
			fcount = fcount + 1	
		x.append(points[icount:(fcount), 0])
		y.append(points[icount:(fcount), 1])
		z.append(points[icount:(fcount), 2])

		
		# print i, icount, fcount
		# print y[i]
		# x.append([])
		# y.append([])
		# z.append([])
		# x[i] = points[icount:fcount-1, 0]
		# y[i] = points[icount:fcount-1, 1]
		# z[i] = points[icount:fcount-1, 2]
		icount = fcount 
		i = i + 1
	# x = np.vstack(x)
	print y
	x = np.array(x)
	y = np.array(y)
	z = np.array(z)
	return x,y,z

# points: list of 3d points
# [[x,y,z],[x,y,z],...,[x,y,z]]
# y represents the axes of the person,
# z the axes of the camera. 
# Assumes same amount of points for every height
def show_surface(points):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	x,y,z = get2Darrays(points)
	# print x
	# print z

	ax.plot_surface(-x,z,y,rstride=1,cstride=1)
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()

# theta = [i/10 for i in xrange(2*32)]
# x = np.reshape([i*10*np.cos(theta) for i in xrange(20)] , -1)
# z = np.reshape([i*10*np.sin(theta) for i in xrange(20)] , -1)
# y = np.reshape([i*np.ones(np.size(theta)) for i in xrange(20)] , -1)
# # print z
# points = [np.array([x[i], y[i], z[i]])for i in xrange(np.size(x))]
# # print points
# # show_pointcloud(points)
# # print points
# show_surface(points)

# receive list of vertix1 vertix2 tuples:
# [([x11,y11,z11],[x21,y21,z21]),...,([x1n,y1n,z1n],[x2n,y2n,z2n])]
def show_lines(vertices):
	fig = plt.figure()
	ax =fig.add_subplot(111,projection='3d')

	for tuples in vertices:
		x1 = tuples[0][0]
		y1 = tuples[0][1]
		z1 = tuples[0][2]
		x2 = tuples[1][0]
		y2 = tuples[1][1]
		z2 = tuples[1][2]
		ax.plot([x1,x2],[y1,y2],[z1,z2])

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	# plt.show()

	return fig,ax

def addpoints(fig,ax,points1,points2 = None):
	points1 = np.array(points1)
	x = points1[:,0]
	z = points1[:,1]
	y = np.zeros((points1.shape[0],1))

	ax.scatter(x,y,z)

	if points2 != None:
		x2 = points2[:,0]
		z2 = points2[:,1]
		y2 = np.zeros((points2.shape[0],1))

		ax.scatter(x2,y2,z2,c='r')


	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()




