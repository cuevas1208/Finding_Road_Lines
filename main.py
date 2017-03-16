## File: main.py
## Name: Manuel Cuevas
## Date: 12/08/2016
## Project: CarND - LaneLines
## Desc: Pipeline uses CV techniques like image threshold, GaussianBlur, Canny,
## and HoughLinesP. 
## Usage: This project identifies a series of steps that identify and draw the
## road lane lines on a from images and video.
## This project was part of the CarND program. 
## Tools learned in class were used to identify lane lines on the road.
#######################################################################################
#importing useful packages
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


print(sys.version)

#reading in an image
image = mpimg.imread('../test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image


# Below are some helper functions from the lesson
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

"""
Applies an image mask.
    
Only keeps the region of the image defined by the polygon
formed from `vertices`. The rest of the image is set to black.
"""
def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

"""
Returns an image with hough lines drawn.
"""
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


# ## Test on Images
# 
# Now you should build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[29]:

import os
imagesList = os.listdir("../test_images/")

for imageLocation in imagesList:
    #open file
    image = mpimg.imread("../test_images/"+imageLocation)
    # Display the image                 
    #plt.figure()
    #plt.imshow(image)
    
    ##################################################################
    ###################COLOR SELECT - WHITE###########################
    # Grab the x and y size and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    gray = np.copy(image)

    # Define color selection criteria
    red_threshold = 180
    green_threshold = 180
    blue_threshold = 180

    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Do a bitwise or with the "|" character to identify the thresholds
    color_thresholds = (image[:,:,0] < rgb_threshold[0]) |                         (image[:,:,1] < rgb_threshold[1]) |                         (image[:,:,2] < rgb_threshold[2])
    gray[color_thresholds] = [0,0,0] 

    # Uncomment the following code if you are running the code locally and wish to save the image
    #mpimg.imsave("test_results/01colorSelection01.png", gray)
    #print ('colorSelection completed')
    
    ##################################################################
    ###################COLOR RANGE - YELLLOW##########################
    # define range of blue color in HSV
    lower_color = np.array([200,190,0])
    upper_color = np.array([255,255,160])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image, lower_color, upper_color)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)

    # merger the greay image and the new image
    color_filter = cv2.bitwise_or(gray, res)
    
    #plt.figure()
    #plt.imshow(color_filter)
    
    ##################################################################
    ###################MASK LINES#####################################
    # Define a triangle region of interest
    left_bottom = [35, ysize]
    right_bottom = [xsize-35, ysize]
    apex = [xsize/2, ysize*(1/2)+45]

    # Fit lines (y=Ax+B) to identify the  3 sided region of interest
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & (YY > (XX*fit_right[0] + fit_right[1])) & (YY < (XX*fit_bottom[0] + fit_bottom[1]))

    # Find where image is both colored right and in the region
    color_filter[~region_thresholds] = [0,0,0]

    # Display our two output images
    #plt.figure()
    #plt.imshow(color_filter) 

    #mpimg.imsave("test_results/region.png", color_filter)
    #print('Mask Completed')
    
    ##################################################################
    ###################Gaussian#######################################
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 7
    blur_gray = cv2.GaussianBlur(color_filter,(kernel_size, kernel_size),0)


    # Define our parameters for Canny and apply
    low_threshold = 50      #wiet, make line think
    high_threshold = 300     #eleminate the background
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Display the image
    plt.figure()
    plt.imshow(edges, cmap='Greys_r')
    #mpimg.imsave("test_results/03"+imageLocation+".png", edges, cmap='Greys_r')
    
    ##################################################################
    ###################Hough transform################################
    # Define the Hough transform parameters
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 120
    max_line_gap = 80
    line_image = np.copy(image)*0 #creating a blank to draw lines on

    # Run Hough on edge detected image and lines draw
    hough = hough_lines(edges, rho, theta, threshold, min_line_length, max_line_gap)

    ##################################################################
    ###################weighted_img################################
    result = weighted_img(hough, image)

    # Save & Display the results
    mpimg.imsave("../test_results/05"+imageLocation+".png", result)
    plt.figure()
    plt.imshow(result) 

