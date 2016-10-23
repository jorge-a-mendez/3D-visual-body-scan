''' ============================================================================================
		
	This file is the implamentation of the main algorithm for the Visual Body Scan.
	Dependencies:
		worldInfo, imgPoints, hullInfo, display3D modules. 
		opencv2, numpy.

    ============================================================================================ ''' 


from __future__ import division	#force / to be float division, // needed for int division
import cv2
from numpy import *
from numpy.linalg import *
import matplotlib as plot
import sys
import worldInfo as wi
import imgPoints as ip
import hullInfo as hull
import display3D as disp
import rotationInfo as rot
from constants import *
import sys

'''By this point, images should all be the same size,
camera should already be calibrated.'''

phone = PHONE_NAME
height = 180
rotating_img_names = IMG_PATH + '/'


# levels: to iterate over the levels of a set of images.
# param: -silhouettes: list of tuples of points, each tuple is a pair of points of
#		  of the silhouette in a level.
#		 -nLevels: number of levels.

def levels(silhouettes,nLevels):
	level = 0
	levels = []
	while level < nLevels:
		for img in silhouettes:
			levels.append(img[level])
		yield levels
		levels = []
		level = level + 1

################################# MAIN LOOP ######################################################################

if len(sys.argv) < 2:
	print "Error!\n\t python " + sys.argv[0] + " [PERSON_HEIGHT centimeters]" 
else:
	height = int(sys.argv[1])
	wi_img = []						#< list of images.
	i = 0
	while(i < 3):					#< Load the images for the vanishing line computation.
		title = INF_LINE_FRAME_FORMAT % (i)
		gray = cv2.imread(title, 0)
		wi_img.append(gray)
		i = i + 1

	# Load calibration information for the specific camera ->
	kmatrix = loadtxt(KMATRIX_FILE)
	world = wi.worldInfo(wi_img,height,kmatrix)
	world.select_heads_feet()   #< this calculates plane at infty and stores it in world.headsplany
								# it also makes it able to compute distance to person.
								# probably should also compute 3D points completely, for intuitiveness

	# Load the set of images for the Visual Body Scan ->
	init = INIT_FRAME
	i = init
	skip = SKIP_FRAMES
	rotating_img = []
	title = rotating_img_names + IMGNAME_FORMAT % init
	color = cv2.imread(title)
	
	while (color != None):
		rotating_img.append(color)
		i = i + skip
		title = rotating_img_names + IMGNAME_FORMAT % i
		color = cv2.imread(title)

		if(i > FINAL_FRAME):
			break

				# Number of images.


	##################################################################################
	# ROTATION COMPUTATION

	orientations = rot.stick(0,rotating_img[0])
	orientations.calibrateFull(rotating_img)
	tot_images = orientations.nValidFrames();

	nLevels = 50						# Number of levels to measure the person.
	imgpoints = ip.imgPoints(nLevels)		# Initialize the object to find silhouttes. 
	
	# Use first image to calibrate the detection engine.
	select_img = rotating_img[0]
	imgpoints.select_threshold(select_img)		 
	imgpoints.select_area(select_img, world)
	headpoint = imgpoints.get_headPoint(select_img,use_distance=False) 		# Get the image point corresponding to the top of the head.

	orig_distance = world.get_DistanceToPerson(headpoint)			# Reference distance to the person.
	# Get the ROI for the future silhouette detection ->
	orig_HeadToNWCorner = world.get_head2PointDistance(headpoint,[imgpoints.orig_ix,imgpoints.orig_iy])
	orig_HeadToSWCorner = world.get_head2PointDistance(headpoint,[imgpoints.orig_ix,imgpoints.orig_fy])
	

	# Detect silhoutte for the rest of the images of the set.
	i = 0
	j = 0
	silhouette = []
	distance_to_person = []
	x_to_person = []
	skip_to_save_results = tot_images//(TOT_IMPOINTS_RESULTS)
	validIndexes = []
	for frame in rotating_img:
		i = i + 1
		rotation = orientations.getRotation(i-1)
		if rotation == None:
			continue
		rotation = degrees(rotation)
		head_top_img = imgpoints.get_headPoint(frame,use_distance=False)
		distance_to_person.append(world.get_DistanceToPerson(head_top_img))
		print 'Frame'
		print i
		x_to_person.append(world.get_XToPerson(head_top_img))

		new_iy = world.get_imgHeightAtHeadDistance(head_top_img,orig_HeadToNWCorner)
		new_fy = world.get_imgHeightAtHeadDistance(head_top_img,orig_HeadToSWCorner)
		sing_silhouette = imgpoints.get_silhouette(new_iy,new_fy, frame)
		

		if sing_silhouette == None:
			# cv2.waitKey(1000)
			continue

		validIndexes.append(i-1)			# valid frames after trying to detect silhouette


################ SHOW IMAGE POINTS###############
			
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		
		# for points in sing_silhouette:
		# 	cv2.circle(gray,points[0],1,255,-1)
		# 	cv2.circle(gray,points[1],1,255,-1)
		# cv2.circle(gray,head_top_img,1,255,-1)
		# cv2.imshow('ImgPoints',gray)
		# edges = imgpoints.__edges__(gray)
		# cv2.imshow('Edges',edges)
		# cv2.waitKey(1000)

#################################################

		silhouette.append(sing_silhouette)	# silhouette: levels x images x 2
		j = j + 1

	cv2.destroyWindow('ImgPoints')
	cv2.destroyWindow('Edges')
	print'validIndexes'
	print validIndexes

	# Visual hull computation for each level of the desired ROI.
	lines = []
	tot_height = abs(orig_HeadToNWCorner - orig_HeadToSWCorner)
	y_jump = tot_height/imgpoints.levels
	y = orig_HeadToNWCorner
	headplane = world.heads_plane
	points = []
	plane = headplane
	plane[0][1] = plane[0][1] + orig_HeadToNWCorner
	center_plane = [array([0,0,0]),plane[1]]	 # plane parallel to the floor that goes through the camera center.
	
	centp_correct_homog = hull.getPlanarHomography(center_plane)


	levels = levels(silhouette,imgpoints.levels)

 	j = 0
 	f = open('./Result_Perimeters.txt','w')
 	f.write('Level,Perimeter\n')
	for level in levels:
		
		i = 0
		plane[0][1] = plane[0][1] + y_jump	# Check sign! It depends on y coordinates growing DOWN
		
		lines = []
		centers = []
		vertices = []
		ROTATION = 0

		# Extract all the rays from the images at the desired level
		for img_points in level:

			# ROTATION = orientations.getRotation(validIndexes[i])	# uncomment to use computed rotations

			# Compute the rays for the image points.
			ray1 = hull.getRay(img_points[0], world.kmtrx)
			ray2 = hull.getRay(img_points[1], world.kmtrx)
			# Compute the lines corresponding to the rays.
			line1 = hull.getLine(ray1, distance_to_person[i], orig_distance, x_to_person[i], ROTATION, centp_correct_homog)
			line2 = hull.getLine(ray2, distance_to_person[i], orig_distance, x_to_person[i], ROTATION, centp_correct_homog)
		
			# For debugging...
			# vertices.append((line1[0][:,0],line1[0][:,0] - 500*line1[1][:,0]))
			vertices.append((line2[0][:,0],line2[0][:,0] - 500*line2[1][:,0]))

			# Project these lines to the plane at the desired level
			line1 = hull.projectLineToPlane(line1,plane)
			line2 = hull.projectLineToPlane(line2,plane)
			lines.append((line1[0],line1[1],line2[1]))
			# Camera centers.
			centers.append(line1[0])
			ROTATION += 2*pi/len(validIndexes)	# uncomment to assume uniform rotation
			i = i + 1
		# Compute the visual hull for the desired level.
 		allpoints,convex,centroid,_ = hull.computeConvexHull(lines)
		points = hull.get3DPoint(allpoints,plane)
		
		perimeter = 0
		p = cv2.convexHull(float32(allpoints))[:,0,:]
		for k in range(p.shape[0]-1):
			print norm(p[k,:]-p[k+1,:])
			perimeter += norm(p[k,:]-p[k+1,:]) 
		# disp.show_pointcloud(points,hull.get3DPoint(centers,plane))
		j += 1	
		f.write('%d,%f\n' % (orig_HeadToNWCorner + y_jump*j,perimeter))

		# disp.show_lines(vertices)