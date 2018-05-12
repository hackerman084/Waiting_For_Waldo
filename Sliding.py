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
    # initializing vars
    best_box = None
    best_box_prob = -np.inf

    # loop window sizes: 20x20, 30x30, 40x40...160x160
    for top in range(0, img.shape[0] - WINDOW_SIZE + 1, step):
        for left in range(0, img.shape[1] - WINDOW_SIZE + 1, step):

            # compute the (top, left, bottom, right) of the bounding box
            box = (top, left, top + WINDOW_SIZE, left + WINDOW_SIZE)

            # crop the original image
            cropped_img = img[box[0]:box[2], box[1]:box[3]]
            # print(cropped_img.shape)
            # print('predicting for box:')
            cv.imshow('IMAGE', cropped_img)

            # cropped_img = cv.resize(cropped_img, dsize=(64, 64), interpolation=cv.INTER_LANCZOS4)

            cropped_img = np.reshape(cropped_img, (1, 64, 64, 3))
            # print(cropped_img.shape)
            # Calls the predict function
            box_prob = predict_function(model, cropped_img)


            first = box_prob[:, 0]
            second = box_prob[:, 1]

            if first > 0.95 and second < 0.05:
                print("I THINK I FOUND A WALDO")
                print(str(box_prob))

                cv.waitKey(0)

            else:
                cv.waitKey(5)

            '''
            if box_prob > best_box_prob:
                best_box = box
                best_box_prob = box_prob
            '''

    return best_box


# Predict function
def predict_function(model, x):
    return model.predict_proba(x)


# This loads the saved trained model that was passed into in argv[2].
trained_model = load_model(model_arg)

# Prints the model summary just for looking at.
trained_model.summary()


get_best_bounding_box(image_path, trained_model)
