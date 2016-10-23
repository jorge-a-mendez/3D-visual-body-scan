# Define the horizontal fraction of the image used as ROI for searching for the silhouette and the head points
NUM_ORIG_IX = 1
DEN_ORIG_IX = 3
NUM_ORIG_FX = 2
DEN_ORIG_FX = 3

# Define the fraction of the person's height used for defining the ROI for searching for the silhouette and head point 
NUM_ORIG_IY = 3
DEN_ORIG_IY = 9
NUM_ORIG_FY = 5
DEN_ORIG_FY = 9

# Define the vertical fraction of the image to disregard in searching for head points
NUM_BORDER_HEAD = 1
DEN_BORDER_HEAD = 5

# Define constants for the rotation information computation. This method is not robust , the values 
FRAC_IMG_ROT_HIGH = .1 	#These first two define the area of the image over which to look or the rotation object (broomstick)
FRAC_IMG_ROT_LOW = .3 	
FRAC_AREAMAX_ROT = .2 	# Fraction of the max_area of the rotation object (broomstick) under which results are discarded

# These two are not currently used. 
FRAC_REG_1 = .4
FRAC_REG_2 = .6

# Define the first and last frames that englobe the whole 360 degree rotations
INIT_FRAME = 36
FINAL_FRAME = 236

# Define the number of frames to skip 
SKIP_FRAMES = 10

# Not currently used
TOT_IMPOINTS_RESULTS = 5

# Define the file paths and names
IMG_PATH = './testimages8'
INF_LINE_FRAME_FORMAT = './testimages8/InfLine%d.jpg'	# Images used for the overlapped iamge
PHONE_NAME = 'OnePlus'
IMGNAME_FORMAT = 'img%.4d.jpg'
KMATRIX_FILE = 'OnePlus_cal.txt'