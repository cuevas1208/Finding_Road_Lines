## File: main.py
## Name: Manuel Cuevas
## Date: 12/12/2016
## Project: CarND - LaneLines
## Desc: Pipeline uses CV techniques like image threshold, GaussianBlur, Canny,
## and HoughLinesP. 
## Usage: This project identifies a series of steps that identify and draw the
## road lane lines on a from images and video.
## This project was part of the CarND program. 
## Tools learned in class were used to identify lane lines on the road.
## Revision: Rev 0000.002 12_10_2016 
#######################################################################################
#importing useful packages
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from drawLineFunctions import *
print(sys.version)

#reading in an image
image = mpimg.imread('../test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image

# # Pipeline
# The output you return should be a color image (3 channel) for processing video below
def pipeline(image):   
    
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
    
    ###################COLOR RANGE - Toget YELLLOW line###############
    # define range of blue color in HSV
    lower_color = np.array([200,190,0])
    upper_color = np.array([255,255,160])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image, lower_color, upper_color)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)

    # merger the greay image and the new image
    color_filter = cv2.bitwise_or(gray, res)
    
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

    ###################Gaussian#######################################
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(color_filter,(kernel_size, kernel_size),0)


    # Define our parameters for Canny and apply
    low_threshold = 50      #wiet, make line think
    high_threshold = 300     #eleminate the background
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    ###################Hough transform################################
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    # NOTE: threshold - it will increase number of intersections needed to detect a 
    # line and as a result reduce number of noise and incorrectly defined lines.
    rho = 2
    theta = np.pi/180
    threshold = 10              
    min_line_length = 125
    max_line_gap = 90

    # Run Hough on edge detected image and lines draw
    hough = hough_lines(edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    ###################weighted_img################################
    #Calculates the weighted sum of two arrays
    weighted_img = cv2.addWeighted(hough, alpha=0.8, src2=image, beta=1., gamma=0.)
    return (weighted_img)


# ## Test on Images
# Build pipeline to work wiht the images in the directory "../test_images"  
import os
imagesList = os.listdir("../test_images/")
print("Processing ", len(imagesList), " images...")
for i, imageLocation in (enumerate(imagesList)):
    #open file
    image = mpimg.imread("../test_images/"+imageLocation)
    
    result = pipeline(image) 
    
    # Save & Display the results
    mpimg.imsave("test_results/05"+imageLocation+".png", result)
    plt.figure()
    plt.imshow(result) 
    print("Images", i, "saved successfully")

print("Program completed :)")

# run your solution on all test_images and make copies into the test_images directory).

# ## Test on Videos
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#Run pipeline with the solid white lane on the right
white_output = 'test_results/white.mp4'
clip1 = VideoFileClip("../test_video/solidWhiteRight.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

#Run pipeline with the solid white lane on the left
yellow_output = 'test_results/yellow.mp4'
clip2 = VideoFileClip('../test_video/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(pipeline)
yellow_clip.write_videofile(yellow_output, audio=False)
