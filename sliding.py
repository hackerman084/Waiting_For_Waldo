import keras
import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import cv2 
import random
# maybe play with making a label array with 2 columns
path = "/Users/AsthaSinghal/Desktop/Hey-Waldo/64/"

WINDOW_SIZE = 64

def get_best_bounding_box(image, model, step=10):
	img = cv2.imread(image)
	#initializing vars
	best_box = None
	best_box_prob = -np.inf

	# loop window sizes: 20x20, 30x30, 40x40...160x160
	for top in range(0, img.shape[0] - WINDOW_SIZE + 1, step):
		for left in range(0, img.shape[1] - WINDOW_SIZE + 1, step):
		# compute the (top, left, bottom, right) of the bounding box
			box = (top, left, top + WINDOW_SIZE, left + WINDOW_SIZE)
			# crop the original image
			cropped_img = img[box[0]:box[2], box[1]:box[3]]
			# predict how likely this cropped image is dog and if higher
			# than best save it
			print('predicting for box %r' % (box, ))
			cropped_img = np.reshape(cropped_img, (1,64,64,3))
			box_prob = predict_function(model, cropped_img)
			print(str(box_prob))
			if box_prob > best_box_prob:
				best_box = box
				best_box_prob = box_prob

	return best_box

def predict_function(model, x):
	return model.predict(x)

get_best_bounding_box("/Users/AsthaSinghal/Desktop/Hey-Waldo/original-images/1.jpg", model)