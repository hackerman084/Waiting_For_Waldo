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
# For every folder passed in as an argument
for index in range(1, len(sys.argv)):
    image_folder_path = sys.argv[index]

    # For every sub-folder in the folder passed
    for folder_name in os.listdir(image_folder_path):
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
                image_dim, _, _ = image.shape
                print(image_dim)

                # print(np.shape(image), '\n')

                # Appending the original images to the x and y datasets
                x_data_set.append(image)
                y_data_set.append(folder_name)

                # ----- IMAGE PROCESSING: -----
                # We do not have enough images of waldo off the bat so
                # we need to make more using various transformations.
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

# Casting the lists as numpy arrays
x_data_set = np.asarray(x_data_set)
y_data_set = np.asarray(y_data_set)

# Doing one-hot-encoding on the labels.
y_data_set = pd.get_dummies(y_data_set)

print("\n")
print("X Data set: ", len(x_data_set), "entries.\tShape: ", x_data_set.shape)
print("Y Data set: ", len(y_data_set), "entries.\tShape: ", y_data_set.shape)
print("\n")
print("---------------------------     FINISHED IMPORTING DATA     ---------------------------\n\n")

# ====== SPLITTING INTO TRAINING AND TESTING DATA ======
x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set, test_size=0.10, random_state=42)


np.random.seed(5)
show_data = 1                   # Variable used to suppress output, 1 = show, 0 = don't show anything
training_time = time.time()        # Starting timestamp, only used to determine how long the program will run


# ====== CREATING THE MODEL ======
model = Sequential()

# Adds a Convolution layer to the model,
model.add(Conv2D(32, (4, 4), padding='same', data_format='channels_first', activation='relu',
                 input_shape=(image_dim, image_dim, 3)))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

# Adding a Dense layer to the model with 100 nodes
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

# This is the last layer of the model and because there are only 2 options:
# "Waldo" and "Not Waldo" there should only be 2 nodes here.
model.add(Dense(2, activation='softmax'))

# ====== HYPER-PARAMETERS ======
batch_size = 100
epochs = 2          # Easy way to change the number of epochs

# ====== COMPILES THE MODEL ======
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=show_data,
          validation_split=0.40,
          shuffle=True,
          # validation_data=(data_file_test, data_labels_test)
          )

# ====== EVALUATING THE MODEL ======
scores = model.evaluate(x_test, y_test,
                        verbose=show_data
                        )

# ====== PRINTING ACCURACY ======
print('%s: %0.6f%%' % (model.metrics_names[1], scores[1]*100))

# ====== SAVING THE KERAS MODEL ======
model.save('waldo_model' + str(image_dim) + '.h5')

# ====== END OF PROGRAM ======

