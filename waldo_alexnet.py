import keras
import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import cv2 
import random

WINDOW_SIZE = 64


path = "/home/justinkterry/waldo/64/"

#will take in an image (as an array) and will randomly add a constant to the rgb values of
#each pixel
#it will then append it to the waldo_data array and augment the waldo_labels array
#this was only really used for waldo_data so 
def color_array(arr, waldo_data, labels): 
	#creating 20 newimages out of 1
	for i in range(1,20):
		#this is the random shift value
		rand = random.randint(1,30)
		#creating separate copies of the shifts to keep it straight
		red_copy = arr.copy()
		blue_copy = arr.copy()
		green_copy = arr.copy()
		rg_copy = arr.copy()
		rb_copy = arr.copy()
		gb_copy = arr.copy()
		rgb_copy = arr.copy()
		#iterate over every pixel in the presumably 64 x 64 image
		for x in range(0,64):
			for y in range(0,64):
				#doing the shifts
				red_copy.itemset((x,y,2), rand)
				blue_copy.itemset((x,y,0), rand)
				green_copy.itemset((x,y,1), rand)
				rg_copy.itemset((x,y,2), rand)
				rg_copy.itemset((x,y,1), rand)	
				rb_copy.itemset((x,y,2), rand)
				rb_copy.itemset((x,y,0), rand)
				gb_copy.itemset((x,y,1), rand)
				gb_copy.itemset((x,y,0), rand)
				rgb_copy.itemset((x,y,0),rand)
				rgb_copy.itemset((x,y,1),rand)
				rgb_copy.itemset((x,y,2),rand)
		#adding it to the waldo data
		waldo_data.append(red_copy)
		waldo_data.append(blue_copy)
		waldo_data.append(green_copy)
		waldo_data.append(rg_copy)
		waldo_data.append(rb_copy)
		waldo_data.append(gb_copy)
		waldo_data.append(rgb_copy)
		#adding it to labels
		labels = labels  [1,1,1,1,1,1,1]
	return labels

#will slide across an image and crop out window_size by window_size images
# will then use the provided model and provided prediction function and use the 
# cropped image as input'
def get_best_bounding_box(image, model, step=10):
	#reading image (file path)
	img = cv2.imread(image)
	#initializing vars(to return the box with the best prediction)
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

#kinda redundant but whatever
def predict_function(model, x):
	return model.predict(x)

#mostly loads the images into an array to later be trained upon
def preprocessing(path, waldo_data, labels):
	for image in os.listdir(path + "waldo"): 
		#because i have so many hidden 
		if (image.endswith("jpg")):
			arr = cv2.imread(path + "waldo/" + image)
			#resizing it to make it compliant with alexnet architecture
			arr = cv2.resize(arr, (224,224))
			#making 200 copies of it 
			for i in range(0,200):
				arr = arr.copy()
				waldo_data.append(arr)
				labels = labels + [1]

	#importing all of the non_waldo images 
	non_waldo = []
	non_waldo_labels = []
	i = len(waldo_data)
	for image in os.listdir(path + "notwaldo"):
		print("NW")
		#only uploading the exact number as waldo_pictures
		if (i > 0):
			if (image.endswith("jpg")):
				arr = cv2.imread(path + "notwaldo/" + image)
				#resize for alexnet
				arr = cv2.resize(arr,(224,224))
				non_waldo.append(arr)
				non_waldo_labels.append(0)
				i = i - 1
		else:
			break
	return waldo_data, labels, non_waldo, non_waldo_labels

# 224 x 244 x 3 --> 11x11 pool 
# 55 x 55 x 96 --> 5x5 pool 
# 27 x 27 x 256 --> 3x3 pool 
# 13 x 13 x 384 --> 3x3 pool 
# 13 x 13 x 256 --> 3x3 
# dense 4096 
# dense 4096 
# dense 1000
#this method models the Alexnet Architecture 
def create_model():
	model = Sequential()
	model.add(Conv2D(96, (55, 55), padding = 'same', activation = "relu", input_shape=(224,224,3)))
	model.add(MaxPooling2D(pool_size=(5, 5)))
	#model.add(Dropout(0.5))
	model.add(Conv2D(256, (27, 27), padding = "same", activation = "relu"))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	#model.add(Dropout(0.5))

	model.add(Conv2D(384, (13, 13), padding = "same", activation = "relu"))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	#model.add(Dropout(0.5))

	model.add(Conv2D(256, (13, 13), padding = "same", activation = "relu"))
	model.add(MaxPooling2D(pool_size=(3, 3))) 

	model.add(Dense(4096, activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Flatten())

	model.add(Dense(1, activation = 'softmax'))

	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
	return model

# returns the numpy array versions of things
def returnDataset(waldo_data, labels, non_waldo, non_waldo_labels):
	x_trainW, x_testW, y_trainW, y_testW = train_test_split(waldo_data,labels, test_size=0.2, random_state=42)
	x_trainNW, x_testNW, y_trainNW, y_testNW = train_test_split(non_waldo,non_waldo_labels, test_size=0.2, random_state=42)

	x_train = x_trainW + x_trainNW
	x_test = x_testW + x_testNW
	y_train = y_trainW + y_trainNW
	y_test = y_testW + y_testNW

	x_train = np.array(x_train)
	x_test = np.array(x_test)
	y_train = np.array(y_train)
	y_test = np.array(y_test)
	return ((x_train, y_train),(x_test, y_test)), x_train

#training the model based off the pretty self-evident parameters
def train_model(waldo_data, labels, non_waldo, non_waldo_labels, model):
	x_trainW, x_testW, y_trainW, y_testW = train_test_split(waldo_data,labels, test_size=0.2, random_state=42)
	x_trainNW, x_testNW, y_trainNW, y_testNW = train_test_split(non_waldo,non_waldo_labels, test_size=0.2, random_state=42)

	x_train = x_trainW + x_trainNW
	x_test = x_testW + x_testNW
	y_train = y_trainW + y_trainNW
	y_test = y_testW + y_testNW

	x_train = np.array(x_train)
	x_test = np.array(x_test)
	y_train = np.array(y_train)
	y_test = np.array(y_test)
	#trying to prevent memory issues
	del x_trainW
	del x_trainNW
	del y_trainW 
	del y_trainNW
	
	#fitting the model
	model.fit(x_train,y_train ,epochs=500,batch_size=128)
	#trying to prevent memory issues
	del waldo_data
	del labels
	del non_waldo
	del non_waldo_labels
	del x_train
	del y_train 
	#testing the model
	scores = model.evaluate(x_test,y_test)
	print("\n%s: %.2f%%" %(model.metrics_names[1], scores[1]))
	return model 

####### Running the model #########
waldo_data = []
labels = []

waldo_dataA, labelsA, non_waldoA, non_waldo_labelsA = preprocessing(path, waldo_data, labels)
model = create_model()
train_model(waldo_dataA, labelsA, non_waldoA, non_waldo_labelsA, model)

#get_best_bounding_box("/Users/AsthaSinghal/Desktop/Hey-Waldo/original-images/1.jpg", model)