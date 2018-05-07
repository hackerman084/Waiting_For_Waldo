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


def color_array(arr, waldo_data, labels): 
	print("B")
	print(len(waldo_data))
	print(len(labels))
	for i in range(1,20):
		#print(i)
		rand = random.randint(1,30)
		red_copy = arr.copy()
		blue_copy = arr.copy()
		green_copy = arr.copy()
		rg_copy = arr.copy()
		rb_copy = arr.copy()
		gb_copy = arr.copy()
		rgb_copy = arr.copy()
		for x in range(0,64):
			for y in range(0,64):
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
		waldo_data.append(red_copy)
		waldo_data.append(blue_copy)
		waldo_data.append(green_copy)
		waldo_data.append(rg_copy)
		waldo_data.append(rb_copy)
		waldo_data.append(gb_copy)
		waldo_data.append(rgb_copy)
		labels = labels + [1,1,1,1,1,1,1]
	print("E")
	print(len(waldo_data))
	print(len(labels))
	return labels
#importing all of the waldo files

import random
import numpy as np

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


# # dummy array of 256X256
# img = np.arange(256 * 256).reshape((256, 256))

# best_box = get_best_bounding_box(img, predict_function)
# print('best bounding box %r' % (best_box, ))


def preprocessing(path, waldo_data, labels):
	for image in os.listdir(path + "waldo"): 
		#print(image + " W")
		arr = cv2.imread(path + "waldo/" + image)
		horiz_img = arr.copy()
		vert_img = arr.copy()
		both_img = arr.copy()
		
		horiz_img = cv2.flip(horiz_img,0)
		vert_img = cv2.flip(vert_img,1)
		both = cv2.flip(both_img, -1)
		h_copy = horiz_img.copy()
		v_copy = vert_img.copy()
		b_copy = both.copy()

		waldo_data.append(arr)
		waldo_data.append(horiz_img)
		waldo_data.append(vert_img)
		waldo_data.append(both)
		labels = labels + [1,1,1,1]

		labels = color_array(arr, waldo_data, labels)

	#importing all of the non_waldo images 
	non_waldo = []
	non_waldo_labels = []
	for image in os.listdir(path + "notwaldo"): 
		#print(image + " NW")
		arr = cv2.imread(path + "notwaldo/" + image)
		#waldo_data.append(arr)
		non_waldo.append(arr)
		non_waldo_labels.append(0)

	return waldo_data, labels, non_waldo, non_waldo_labels


def create_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(64,64,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.15))
	model.add(Dense(1))
	model.add(Activation('softmax'))

	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
	return model



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

	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)

	model.fit(x_train, y_train,epochs=1,batch_size=200)
	scores = model.evaluate(x_test,y_test)
	print("\n%s: %.2f%%" %(model.metrics_names[1], scores[1]))
	return model 


waldo_data = []
labels = []

waldo_dataA, labelsA, non_waldoA, non_waldo_labelsA = preprocessing("/Users/AsthaSinghal/Desktop/Hey-Waldo/64/", waldo_data, labels)
model = create_model()
train_model(waldo_dataA, labelsA, non_waldoA, non_waldo_labelsA, model)

get_best_bounding_box("/Users/AsthaSinghal/Desktop/Hey-Waldo/original-images/1.jpg", model)









