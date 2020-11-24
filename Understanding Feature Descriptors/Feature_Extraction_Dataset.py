import tensorflow
from PIL import Image
import numpy as np
from skimage.feature import hog, local_binary_pattern
import cv2
import matplotlib.pyplot as plt
import pandas as pd
feature = []
img_list = ["98.jpg", "174.jpg", "212.jpg"]
for img_name in img_list:
	img = cv2.imread(img_name)
	img_grey = np.array(Image.open(img_name).convert('LA'))
	img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	
	#Splitting into channels!
	img_color = tensorflow.image.resize(img_color, (256,256),preserve_aspect_ratio=False).numpy()
	img_r = np.reshape(img_color[:, :, 0], (256,256))
	img_g = np.reshape(img_color[:, :, 1], (256,256))
	img_b = np.reshape(img_color[:, :, 2], (256,256))
	
	img_grey = tensorflow.image.resize(img_grey, (256,256),preserve_aspect_ratio=False).numpy()
	img_grey = np.reshape(img_grey[:, :, 0], newshape=(256,256))
	#
	# img_hsv = tensorflow.image.resize(img_hsv, (256,256),preserve_aspect_ratio=False).numpy()
	img_h = np.reshape(img_hsv[:, :, 0], (256,256))
	img_s = np.reshape(img_hsv[:, :, 1], (256,256))
	img_v = np.reshape(img_hsv[:, :, 2], (256,256))
	#
	plt.figure(figsize = (16,16))
	radius = 32
	n_points = 8 * radius
	lbp_grey = local_binary_pattern(img_grey, n_points, radius)
	lbp_r = local_binary_pattern(img_r, n_points, radius)
	lpg_v = local_binary_pattern(img_v, n_points, radius)
	
	#plt.imshow(lbp_image)
	plt.subplot(122)
	fd_lbphog_grey, lbp_hog = hog(lbp_grey, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True)
	plt.imshow(lbp_hog)
	fd_lbphog_color, lbp_hog = hog(lbp_r, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True)
	fd_lbphog_hsv, lbp_hog = hog(lbp_v, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True)
	
	array = np.concatenate([fd_lbphog_grey, fd_lbphog_color, fd_lbphog_hsv], axis = 0)
	features.append(array)
	
features = pd.DataFrame(features)
features.to_csv("Features_Sample.csv", sep=',')

