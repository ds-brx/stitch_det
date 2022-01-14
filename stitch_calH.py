import cv2
import numpy as np
#from numba import jit,njit,prange

alpha_saved = np.load('alpha.npy')
beta_saved = np.load('beta.npy')
def stitch_by_H(img1, img2, H, frame_number):
	"""Use the key points to stitch the images.
	img1: the image containing frames that have been joint before.
	img2: the newly selected key frame.
	H: Homography matrix, usually from compute_homography(img1, img2).
	"""

	# cv2.imshow("Image 1", img1)
	# cv2.waitKey()
	# cv2.imshow("Image 2", img2)
	# cv2.waitKey()
	# Get heights and widths of input images
	h1, w1 = img1.shape[0:2]
	h2, w2 = img2.shape[0:2]
	# print(frame_number)

	# Store the 4 ends of each original canvas
	img1_canvas_orig = np.float32([[0, 0], [0, h1],
								   [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
	img2_canvas = np.float32([[0, 0], [0, h2],
							  [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

	# The 4 ends of (perspective) transformed img1
	img1_canvas = cv2.perspectiveTransform(img1_canvas_orig, H)

	# Get the coordinate range of output (0.5 is fixed for image completeness)
	output_canvas = np.concatenate((img2_canvas, img1_canvas), axis=0)
	[x_min, y_min] = np.int32(output_canvas.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(output_canvas.max(axis=0).ravel() + 0.5)

	# The output matrix after affine transformation
	transform_array = np.array([[1, 0, -x_min],
								[0, 1, -y_min],
								[0, 0, 1]])

	# Warp the perspective of img1
	img_output = cv2.warpPerspective(img1, transform_array.dot(H),
									 (x_max - x_min, y_max - y_min))

	
	# This code does what the final_warp function was doing where we initially
	# Had used numba ,try commenting out next to lines to see change in O/P
	# copy = img_output[-y_min:h2 - y_min,-x_min:w2-x_min]
	# blend1 = copy[:, 0:(copy.shape[1]//5) + 1]
	# blend2 = img2[:, 0:(copy.shape[1]//5) + 1]
	# #print(frame_number)
	# if frame_number == 1:
	# 	ramp = np.linspace(1, 0, blend2.shape[1])
	# 	ramp = np.tile(np.transpose(ramp), (blend2.shape[0], 1))
	# 	# print(copy.shape[1], blend2.shape[0])

	# 	ramp = cv2.merge([ramp,ramp,ramp,ramp])

	# 	np.save('alpha.npy',ramp)
	# 	np.save('beta.npy',(1 - ramp))
	# alpha = alpha_saved #np.load('alpha.npy')
	# beta = beta_saved #np.load('beta.npy')

	# # blended = ramp * blend1 + (1 - ramp) * blend2
	# blended = alpha * blend1 + beta * blend2
	# # blended = np.clip((blended), 0, 255)
	# # blended = np.uint8(blended)
	# img2[:, 0:(copy.shape[1]//5) + 1] = blended
	img_output[-y_min:h2 - y_min,-x_min:w2-x_min] = img2
	return img_output


# Inspired from http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

def compute_homography(img1, img2): # cant jit this function as not only numpy is upscaled 
	"""Find SIFT matches and return the estimated Homography matrix."""
	# Call the SIFT method
	#sift = cv2.ORB_create()

	h1, w1 = img1.shape[0:2]
	h2, w2 = img2.shape[0:2]

	img1_crop = img1[:, w1 - w2:]  # Crop the right part of img1 for detecting SIFT
	diff = np.size(img1, axis=1) - np.size(img1_crop, axis=1)

	fast = cv2.FastFeatureDetector_create(threshold = 15) # Thresholding to eliminate spurious keypoints
	kps,features = None,None
	brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes = 32)  #A fastest variant

	kp1 = fast.detect(img1_crop,None) #B
	kp1,des1 = brief.compute(img1_crop,kp1)  #C Comment out this line to do for only fast

	kp2 = fast.detect(img2,None) #B
	kp2,des2 = brief.compute(img2,kp2)

	# Use the Brute-Force matcher to obtain matches
	bf = cv2.BFMatcher(cv2.NORM_HAMMING)  # Using L2 (Euclidean) distance
	matches = bf.knnMatch(des1, des2, k=2)
	matches = np.array(matches)

	# Define a Valid Match: whose distance is less than match_ratio times the
	# distance of the second best nearest neighbor.
	match_ratio = 0.6 # I can JIT this part also to make it faster

	# Pick up valid matches
	valid_matches = []
	for m1, m2 in matches:				# Does this really make sense for ORB
		if m1.distance < match_ratio * m2.distance:
			valid_matches.append(m1)

	min_match_num = 4  # Minimum number of matches (to ensure a good stitch)
	if len(valid_matches) > min_match_num:
		# Extract the coordinates of matching points
		img1_pts = []
		img2_pts = []
		for match in valid_matches:
			img1_pts.append(kp1[match.queryIdx].pt)
			img2_pts.append(kp2[match.trainIdx].pt)

		# Formalize as matrices (for the sake of computing Homography)
		img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
		img1_pts[:, :, 0] += diff  # Recover its original coordinates
		img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

		# Compute the Homography matrix
		H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

		return H
	else:
		# Print a warning message, but compute Homography anyway as long as
		# len(valid_matches) > 4, which is the least requirement of computing
		# Homography.
		pass

# We are not using this function anymore due to numba dependency
#@jit(cache= True,parallel=True)
# @jit(cache= True,parallel=True)
def warp_blend(img_output,img2,x_min,x_max,y_min,y_max,h2,w2):
	alpha = 0.8
	for i in prange(-y_min, h2 - y_min):
		for j in range(-x_min, w2 - x_min):
			if np.any(img2[i + y_min][j + x_min]):
				if np.any(img_output[i][j]):
					img_output[i][j] = (alpha * img2[i + y_min][j + x_min]+ (1 - alpha) * img_output[i][j])
				else:
					img_output[i][j] = img2[i + y_min][j + x_min]

	return img_output

def stitch_image(img1, img2, frame_number, temp=None,i=None):
	# Input images
	if isinstance(img1, str):
		img1 = cv2.imread(img1)

	if(i==1):
		img1 = cv2.resize(img1, (854,480)) # Resize to 480p for faster computation
		img1 = cylindrical_project(img1,temp) # Project to Cylinder ,our preprocessing step
		img1 = crop_black(img1) # We get some black patches on left and right side remove them
	# cv2.imshow("img1",img1)
	# cv2.waitKey()
	if isinstance(img2, str):
		img2 = cv2.imread(img2)
	
	img2 = cv2.resize(img2, (854,480))
	# cv2.imshow("img2",img2)
	# cv2.waitKey()

	# Apply cylindrical projection to the new frame
	img2 = cylindrical_project(img2,temp)
	img2 = crop_black(img2)
	H = compute_homography(img1, img2)
	if(i==1):
		np.save('i1i2.npy',H)

	if(i==2):
		np.save('i2i3.npy',H)

	if(i==3):
		np.save('i3i4.npy',H)

	# Load the precomputed Homography matrix for respective
	# Adjacent frames, these matrix ones calculated is same 
	# For rest of the future
	# if(i==1):
	# 	H = np.load('i1i2.npy')

	# if(i==2):
	# 	H = np.load('i2i3.npy')

	# if(i==3):
	# 	H = np.load('i3i4.npy')

	img_output = stitch_by_H(img1, img2, H, frame_number)
	h1, w1 = img1.shape[0:2]
	h2, w2 = img2.shape[0:2]
	#stitched_image = warp_blend(img_output,img2,x_min,x_max,y_min,y_max,h2,w2)
	stitched_image = crop_black(img_output)

	return stitched_image


# Took this from StackOverflow to crop black patches
def crop_black(img):
	"""Crop off the black edges."""
	max_area = 1
	best_rect = None
	#print(img.shape)
	if(img.shape == (480,854,4)):
		#print("Case 1")
		best_rect = (56,0,743,480)
	elif(img.shape == (501,1055,4)):
		#print("Case 2")
		best_rect = (1,0,1054,500)

	elif(img.shape == (588, 1138, 4)):
		best_rect = (4, 28, 1134, 493)

	elif(img.shape == (570, 1096, 4)):
		best_rect = (0, 27, 1096, 502)
		
	# elif(img.shape == (564,1096,4)):
	# 	#print("Case 3")
	# 	best_rect = (0,27,1096,500)
	# elif(img.shape == (575,1137,4)):
	# 	#print("Case 4")
	# 	best_rect = (3,28,1134,490)

	

	else:
		#print("best_rect was None")
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_, thresh = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
		contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
										  cv2.CHAIN_APPROX_NONE)

		max_area = 0
		best_rect = (0, 0, 0, 0)

		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)
			deltaHeight = h - y
			deltaWidth = w - x

			area = deltaHeight * deltaWidth

			if area > max_area and deltaHeight > 0 and deltaWidth > 0:
				max_area = area
				best_rect = (x, y, w, h)

		# print(img.shape)
		# print(best_rect)

	# print(img.shape)
	# print(best_rect)
	if max_area > 0:
		img_crop = img[best_rect[1]:best_rect[1] + best_rect[3],
				   best_rect[0]:best_rect[0] + best_rect[2]]
	else:
		img_crop = img

	return img_crop


def cylindrical_project(img,temp = None):
	"""This function returns the cylindrical warp for a given image and intrinsics matrix K"""
	if(temp is None):
		h_,w_ = img.shape[:2]
		K = np.array([[600,0,w_/2],[0,600,h_/2],[0,0,1]])
		# pixel coordinates
		y_i, x_i = np.indices((h_,w_))
		X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
		Kinv = np.linalg.inv(K) 
		X = Kinv.dot(X.T).T # normalized coords
		# calculate cylindrical coords (sin\theta, h, cos\theta)
		A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
		B = K.dot(A.T).T # project back to image-pixels plane
		# back from homog coords
		B = B[:,:-1] / B[:,[-1]]
		# make sure warp coords only within image bounds
		B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
		B = B.reshape(h_,w_,-1)
		temp = B
	
	#np.save('nxx.npy',B)
	B = temp
	img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
	# warp the image according to cylindrical coords
	return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA)
