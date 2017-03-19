## File: main.py
## Name: Manuel Cuevas
## Date: 12/08/2016
## Project: CarND - LaneLines
## Desc: Pipeline uses CV techniques like image threshold, GaussianBlur, Canny,
## and HoughLinesP.
## Usage: This project identifies a series of steps that identify and draw the
## road lane lines from images and video.
## This project was part of the CarND program.
## Tools learned in class were used to identify lane lines on the road.
## Revision: Rev 0000.004 12_11_2016
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

# # Pipeline
# The output you return should be a color image (3 channel) for processing video below
def pipeline(image):   

    ###################Color and Mask#################################
    #Converts an image from one color space to another
    color_filter = colorTransformation(image)

    #Mask the image to obtain the image area to work  
    mask_filter = maskImage(image, color_filter)
    
    ###################Gaussian#######################################
    # Define a kernel size and apply Gaussian smoothing
    # ksize.width and ksize.height can differ
    kernel_size = 5  # must be positive and odd
    blur_gray = cv2.GaussianBlur(color_filter,(kernel_size, kernel_size),0)

    ##################Canny###########################################
    #Define our parameters for Canny and apply
    #Canny finds edges in the input image and marks them in the output map edges
    low_threshold = 50       #made lines think
    high_threshold = 300     #eleminate the background
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    ##################Hough transform#################################
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    # NOTE: threshold - it will increase the number of intersections needed to detect a
    # line and as a result reduce number of noise and incorrectly defined lines.
    rho = 2
    theta = np.pi/180
    threshold = 10              
    min_line_length = 125
    max_line_gap = 90

    # Run Hough on edge detected image and lines draw
    hough = hough_lines(edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    ###################weighted_img###################################
    #Calculates the weighted sum of two arrays
    weighted_img = cv2.addWeighted(hough, alpha=0.8, src2=image, beta=1., gamma=0.)
    return (weighted_img)


### Test on Images
# Build pipeline to work with the images in the directory "test_images/"  
import os
imagesList = os.listdir("test_images/")
print("Processing ", len(imagesList), " images...")
for i, imageLocation in (enumerate(imagesList)):
    #open file
    image = mpimg.imread("test_images/"+imageLocation)
    
    result = pipeline(image)
    
    # Save & Display the results
    mpimg.imsave("test_results/05"+imageLocation+".png", result)
    plt.figure()
    plt.imshow(result)
    print("Images", i, "saved successfully")

### Test on Videos
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

print("Program completed :)")


