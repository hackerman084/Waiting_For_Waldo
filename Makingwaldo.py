import pandas as pd
import numpy as np
import os
import cv2 as cv
import random
import sys


# PATHS
# Reads the image passed by user, expecting a large image.
original_image_name = sys.argv[1]

# gives the path of the location this file is running in
file_path = os.path.dirname(os.path.realpath('__file__'))
print('file path :', file_path)

# Path to picture:
path_to_picture = os.path.join(file_path, 'original-images', original_image_name)
print('path to picture', path_to_picture)


# Reads the image:
image = cv.imread(path_to_picture)

# Print dimensions
print(image.shape)


# The location of waldo's head is taken in as the second and third argument
# NOTE: the indexing is [y,x] not [x,y]
center_head_y = int(sys.argv[2])
center_head_x = int(sys.argv[3])

# With these numbers 44 and 64 and there should 20x20 more pictures per single pictures.
HEAD_BOUNDS = 44
WINDOW_DIMENSION = 64


# finding the coordinates of the head bounding box:
y_head_top = int(center_head_y - (0.5 * HEAD_BOUNDS))
y_head_bottom = int(center_head_y + (0.5 * HEAD_BOUNDS))

x_head_left = int(center_head_x - (0.5 * HEAD_BOUNDS))
x_head_right = int(center_head_x + (0.5 * HEAD_BOUNDS))

# Locations of the bounding coordinate of the cropping frames:
y_min_crop = y_head_bottom - WINDOW_DIMENSION
y_max_crop = y_head_top + WINDOW_DIMENSION

x_min_crop = x_head_right - WINDOW_DIMENSION
x_max_crop = x_head_left + WINDOW_DIMENSION

# Now scan through the picture and get every top left corner of the window that satisfies the conditions above.
# if the upper left corner is inside the bounding box we know that the window of size 64x64 contains waldo's head.
# we will then save that image of waldo to use for training.

# calculating the cropping box, the box that contains waldos head in the bottom right corner.
x_i = 0
while x_i < WINDOW_DIMENSION - HEAD_BOUNDS + 1:
    y_i = 0
    while y_i < WINDOW_DIMENSION - HEAD_BOUNDS + 1:

        start_y = y_min_crop + y_i
        end_y = y_head_bottom + y_i

        start_x = x_min_crop + x_i
        end_x = x_head_right + x_i

        crop_img = image[start_y:end_y, start_x:end_x]
        cv.imshow("cropped", crop_img)
        cv.waitKey(10)

        # print(y_i, x_i, crop_img.shape)

        # Saving the new cropped picture into a waldo folder
        cropped_picture_name = os.path.basename(os.path.splitext(original_image_name)[0]) +\
                               '_' + str(y_i) + '_' + str(x_i) + '.jpg'
        # print(cropped_picture_name)

        path_to_cropped_image = os.path.join(file_path, '64BetterWaldos/waldo', cropped_picture_name)
        print(path_to_cropped_image)

        cv.imwrite(path_to_cropped_image, crop_img)
        y_i = 1 + y_i

    x_i = 1 + x_i



#
'''
print('Y Cropping Range: ', start_y, ' to ', end_y)
print('X Cropping Range: ', start_x, ' to ', end_x)
'''



'''
crop_img = image[y_start:y_start + WINDOW_DIMENSION, x_start:x_start + WINDOW_DIMENSION]
cv.imshow("cropped", crop_img)
cv.waitKey(0)
'''




