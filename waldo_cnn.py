"""
Written by Elliot Hall, Astha Singhal and Shruti Sharma
This code was written for the final project for PHYS476.

====== NOTES ======
This program will accept multiple folders containing categorical sub-folders, in the case of the final project,
the sub folders are "waldo" and "not waldo". The program will read in all of those pictures into 2 lists
the first list contains square pictures in the folder.
The second list contains an equal number of labels. The label is the name of the folder by default.

All pictures submitted to the training program need to be the same size and square.

The labels are one-hot-encoded
The program is trained and the Keras model is exported.
"""

# ====== IMPORTS ======
import sys              # imports the system for passing command line arguments
import os               # imports the os stuff
import time             # imports time to do timing stuff
import cv2 as cv        # imports OpenCV as cv
import numpy as np      # imports numpy
import pandas as pd     # imports pandas for data frame use
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils
from keras.models import load_model
import keras
import tensorflow


# ----- DATA PATH: -----
# gives the path of the location this file is running in
file_path = os.path.dirname(os.path.realpath('__file__'))

# the argument passed is the folder containing the images.
# the folder passed must be a direct offspring of the folder containing this piece of code.

# All appending will be done in commutable forms,
# This is to prevent copying of the entire set multiple times and slowing down the process.

# Empty list that will fill up with the image arrays
x_data_set = []

# Empty list that will fill up with the folder names
y_data_set = []


# ====== DATA IMPORTATION ======
# The Following nested for loops are iterating through all of the data in all of the folders and creating a numpy array
# That will contain each picture and its label.
file_count = 0
index = 1
image_folder_path = sys.argv[1]
# For every folder passed in as an argument
for index in range(1, len(sys.argv)):
    image_folder_path = sys.argv[index]

    # For every sub-folder in the folder passed
    for folder_name in os.listdir(image_folder_path):

        # image_type_folder_path is the waldo or not waldo folders.
        image_type_folder_path = os.path.join(image_folder_path, folder_name)

        print('-------------------------------', folder_name, '-------------------------------')

        # For each file in the folder:
        for filename in os.listdir(image_type_folder_path):

            # The file extension that will be used to check the type of file it is.
            temp_file_ext = os.path.splitext(filename)[1]

            # If the file is a jpg file extension then we will use it,
            # this just gets any random file types out of the way.
            if temp_file_ext == '.jpg' or ".jpeg":
                file_count = file_count + 1

                # the location of each file as used by the OpenCV reading method.
                image_path = os.path.join(image_type_folder_path, filename)
                print('%5d' % file_count, '\t', image_path, "  \t", end="")

                # reads the file as a numpy array in the form of a matrix with the
                image = cv.imread(image_path)

                # Converts image to grey scale.
                # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                # Normalizes the values of the RGB color spaces to values between 0 and 1.
                image = image/255

                # Gets the square image dimensions to use later.
                image_dim, _, _ = image.shape
                print(image_dim)

                # print(np.shape(image), '\n')

                # Appending the original images to the x and y data sets
                x_data_set.append(image)
                y_data_set.append(folder_name)

                # ----- IMAGE PROCESSING: -----
                # We do not have enough images of waldo off the bat so
                # we need to make more using various transformations and rotations
                # each image in the waldo image folder becomes 8 images after this convolution
                # the original image + 7 altered images.
'''
                if folder_name == 'waldo':
                
                    # Rotating +90, +180, +270
                    r = 1
                    print("\t\t", end="")
                    while r < 3:
                        image_R = np.rot90(image, k=r, axes=(0, 1))
                        x_data_set.append(image_R)
                        y_data_set.append(folder_name)
                        print(r, end=" ")
                        r = r + 1
                        file_count = file_count + 1

                    # Flipping Horizontally
                    image_H = np.flip(image, axis=0)
                    x_data_set.append(image_H)
                    y_data_set.append(folder_name)
                    print(4, end=" ")
                    file_count = file_count + 1

                    # Rotating the Flipped image +90
                    image_H_90 = np.rot90(image_H, k=1, axes=(0, 1))
                    x_data_set.append(image_H_90)
                    y_data_set.append(folder_name)
                    print(5, end=" ")
                    file_count = file_count + 1

                    # Flipping Vertically
                    image_V = np.flip(image, axis=1)
                    x_data_set.append(image_V)
                    y_data_set.append(folder_name)
                    print(6, end=" ")
                    file_count = file_count + 1

                    # Rotating the Flipped image +90
                    image_V_90 = np.rot90(image_V, k=1, axes=(0, 1))
                    x_data_set.append(image_V_90)
                    y_data_set.append(folder_name)
                    print(7)
                    file_count = file_count + 1
                    
            else:
                continue
'''
# Casting the lists as numpy arrays
x_data_set = np.asarray(x_data_set)
y_data_set = np.asarray(y_data_set)

# Doing one-hot-encoding on the labels.
y_data_set = pd.get_dummies(y_data_set)


# print(y_data_set)


print("\n")
print("X Data set: ", len(x_data_set), "entries.\tShape: ", x_data_set.shape)
print("Y Data set: ", len(y_data_set), "entries.\tShape: ", y_data_set.shape)
print("\n")
print("---------------------------     FINISHED IMPORTING DATA     ---------------------------\n\n")


# ====== SPLITTING INTO TRAINING AND TESTING DATA ======
x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.33, random_state=42)

print("X Train Data set: ", len(x_train), "entries.\tShape: ", x_train.shape)
print("Y Train Data set: ", len(y_train), "entries.\tShape: ", y_train.shape)
print('\n')
print("X Test Data set: ", len(x_test), "entries.\tShape: ", x_test.shape)
print("Y Test Data set: ", len(y_test), "entries.\tShape: ", y_test.shape)

# Random seed
np.random.seed(42)


show_data = 1                       # Variable used to suppress output, 1 = show, 0 = don't show anything

training_time = time.time()         # Starting timestamp, only used to determine how long the program will run


# ====== CREATING THE MODEL ======
# Model parameters are based off of AlexNet, however the actual dimensions are different
# to accommodate our differing image sizes.
# Dropout layers are used to minimize over-fitting in this network,
# The data set is very repetitive and it would be very easy to brute-force toward the solution without
# any learning taking place.
model = Sequential()

# Layer 1
Convolution_Layer_1 = model.add(Conv2D(32, (16, 16), strides=(2, 2), padding='same',
                                       data_format='channels_first', activation='relu',
                                       input_shape=(image_dim, image_dim, 3)))
Max_Pooling_Layer_1 = model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
Convolution_Layer_2 = model.add(Conv2D(48, (8, 8), strides=(1, 1), activation='relu'))
Max_Pooling_Layer_2 = model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
Convolution_Layer_3 = model.add(Conv2D(64, (4, 4), strides=(1, 1), activation='relu'))
Flatten_Layer_1 = model.add(Flatten())
Dropout_Layer_1 = model.add(Dropout(0.25))

# Layer 4
Dense_Layer_1 = model.add(Dense(1000, activation='relu'))

# Layer 5
Dense_Layer_2 = model.add(Dense(500, activation='relu'))
Dropout_Layer_2 = model.add(Dropout(0.15))

# Layer 6
# This is the last layer of the model and because there are only 2 options;
# "Waldo" and "Not Waldo" there are only 2 nodes here.
Dense_Layer_3 = model.add(Dense(2, activation='softmax'))


# ====== HYPER-PARAMETERS ======
# There was a problem with memory consumption having to do with the batch size.
# batch_size appears to be inversely proportional to the memory used. High batch = low memory usage.
batch_size = 50
epochs = 30         # A number between 20 to 30 produces a high accuracy model that is not over-trained.


# ====== Optimizer Values ======
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)


# ====== COMPILES THE MODEL ======
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

# ====== PRINTS THE MODEL COMPOSITION SUMMARY ======
model.summary()


# ====== TRAINS THE MODEL ======
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=show_data,
          validation_split=0.33,
          shuffle=True,
          # validation_data=(x_test, y_test)
          )


# ====== EVALUATING THE MODEL ======
scores = model.evaluate(x_test, y_test, verbose=show_data)


# ====== PRINTING ACCURACY ======
print('%s: %0.6f%%' % (model.metrics_names[1], scores[1]*100))


# ====== SAVING A DIAGRAM OF THE MODEL TO FILE ======
model_name = 'waldo_model-' + image_folder_path + '-' + str(round(scores[1]*100))

keras.utils.plot_model(model,
                       to_file=model_name + '.png',
                       show_shapes=False,
                       show_layer_names=True,
                       rankdir='LR')


# ====== PRINTING A SUMMARY OF THE MODEL WEIGHTS AND BIASES ======
# keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)


# ====== SAVING THE KERAS MODEL ======
model.save(model_name + '.h5')
print("Saved as: " + model_name + '.h5')

# ====== END OF PROGRAM ======


