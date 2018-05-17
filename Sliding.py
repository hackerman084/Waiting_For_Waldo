import keras
import pandas as pd
import numpy as np
import os
import cv2 as cv
import random
import sys

from keras.models import load_model
# maybe play with making a label array with 2 columns
# When using this copy the path to the picture you want to search in the first slot
# and the name of the model in the second slot.

# A NOTE ON MODEL NAMING CONVENTION:
# The model has a name waldo_model-(images trained on)-(final accuracy).h5

image_path = sys.argv[1]
model_arg = sys.argv[2]

WINDOW_SIZE = 64


def get_best_bounding_box(image, model, step=4):
    img = cv.imread(image)
    y = img.shape[0]
    x = img.shape[1]

    # Create a black image




    cv.imshow('WORK YOU FUCK!', mask)
    cv.waitKey(0)
    
    # initializing vars
    # best box does not work because the neural network is too well trained on very specific examples of waldo.

    # loop window sizes: 20x20, 30x30, 40x40...160x160
    # current functionality only allows for 64x64 windows.
    for top in range(0, img.shape[0] - WINDOW_SIZE + 1, step):
        for left in range(0, img.shape[1] - WINDOW_SIZE + 1, step):

            # compute the (top, left, bottom, right) of the bounding box
            box = (top, left, top + WINDOW_SIZE, left + WINDOW_SIZE)

            # crop the original image
            cropped_img = img[box[0]:box[2], box[1]:box[3]]
            # print(cropped_img.shape)
            # print('predicting for box:')

            # Doing masking stuff:
            mask = np.zeros((y, x, 3), np.uint8)
            cv.rectangle(mask, (0, 0), (128, 128), (255, 255, 255), -1)

            # shows the current window that is being looked at by the model.
            cv.imshow('IMAGE', img)

            # cropped_img = cv.resize(cropped_img, dsize=(64, 64), interpolation=cv.INTER_LANCZOS4)

            # Reshapes the image into the correct shape to be fed into the model
            cropped_img = np.reshape(cropped_img, (1, 64, 64, 3))

            # Calls the predict function
            box_prob = predict_function(model, cropped_img)

            # These are the individual values returned by box_prob.
            # NOTE: the values are stored in a column vector.
            first = box_prob[:, 1]
            second = box_prob[:, 0]

            # Checking to confirm that the vector received by box_prob suggests that there is a waldo.
            # NOTE: The expected value for a waldo is [0,1]
            # this is confirming that the model is 95% sure that it has found a waldo.
            if first < 0.5 and second > 0.95:
                print("I THINK I FOUND A WALDO")
                print(str(box_prob))

                # Waits for user keystroke.
                cv.waitKey(0)

            else:
                # Waits 5ms then moves on.
                cv.waitKey(5)


# Predict function
def predict_function(model, x):
    return model.predict(x)


# This loads the saved trained model that was passed into in argv[2].
trained_model = load_model(model_arg)

# Prints the model summary just for looking at.
trained_model.summary()


get_best_bounding_box(image_path, trained_model)



