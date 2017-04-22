## File: basic.py
## Name: Manuel Cuevas
## Date: 12/08/2016
## Project: CarND - LaneLines
## Desc: Pipeline uses CV techniques like image threshold, GaussianBlur, Canny,
## and HoughLinesP.
## Usage: This project follows a series of steps to identify and draw the
## road lane lines from images and video.
#######################################################################################
#importing useful packages
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import glob
from drawLineFunctions import *


class basicLine:
    ''' Pipeline
    The output you return should be a color image (3 channel) for processing video below
    Input: image - image to be process
           print_Image - to print pipeline images
    Output: Original image with the the detected road line
    '''
    def pipeline(self, image, print_Image = False):   

        ###################Color and Mask#################################
        #Converts an image from one color space to another
        color_filter = colorTransformation(image)
        if print_Image:
            mpimg.imsave("test_results/00color_filter.png", color_filter)

        #Mask the image to obtain the image area to work  
        mask_filter = maskImage(image, color_filter)

        #Uncomment the following code if you wish to save the image
        if print_Image:
            mpimg.imsave("test_results/01mask_filter.png", mask_filter)
        ###################Gaussian#######################################
        # Define a kernel size and apply Gaussian smoothing
        # ksize.width and ksize.height can differ
        kernel_size = 5  # must be positive and odd
        blur_gray = cv2.GaussianBlur(color_filter,(kernel_size, kernel_size),0)

        #Uncomment the following code if you wish to save the image
        if print_Image:
            mpimg.imsave("test_results/02GaussianBlur.png", blur_gray)
        ##################Canny###########################################
        #Define our parameters for Canny and apply
        #Canny finds edges in the input image and marks them in the output map edges
        low_threshold = 50       #made lines think
        high_threshold = 300     #eleminate the background
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        #Uncomment the following code if you wish to save the image
        if print_Image:
            mpimg.imsave("test_results/03Edges.png", edges)
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
        
        #Uncomment the following code if you wish to save the image
        if print_Image:
            mpimg.imsave("test_results/04Hough.png", hough)
        ###################weighted_img###################################
        #Calculates the weighted sum of two arrays
        weighted_img = cv2.addWeighted(hough, alpha=0.8, src2=image, beta=1., gamma=0.)

        #Uncomment the following code if you wish to save the image
        if print_Image:
            mpimg.imsave("test_results/05weighted_img.png", weighted_img)
        return (weighted_img)
