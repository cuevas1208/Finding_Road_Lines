## File: main.py
## Name: Manuel Cuevas
## Date: 12/08/2016
## Project: CarND - LaneLines
## Desc: Pipeline uses CV techniques like image threshold, GaussianBlur, Canny,
## and HoughLinesP.
## Usage: This project follows a series of steps to identify and draw the
## road lane lines from images and video.
## This project was part of the CarND program.
## Tools learned in class were used to identify lane lines on the road.
## Revision: Rev 0000.004 12_11_2016
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
print(sys.version)

''' Pipeline
The output you return should be a color image (3 channel) for processing video below
Input: image - image to be process
       print_Image - to print pipeline images
Output: Original image with the the detected road line
'''
def pipeline(image, print_Image = False):   

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

''' main
Saves the pipeline output in to test_results directory
Input: Directory location of the videos to process
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Arguments')
    parser.add_argument(
        'video_location',
        type=str,
        nargs='?',
        default='',
        help='Path to video folder. This is where the video will be loaded from'
    )
    args = parser.parse_args()

    #Create output directory 
    if not os.path.exists("test_results"):
        os.makedirs("test_results")

    ### Test on Images
    # Build pipeline to work with the images in the directory "test_images/"
    if os.path.exists("test_images/"):
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
    else:
        print("Missing images dirctory test_images")

    
    ### Test on Videos
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML

    if args.video_location != '':
        #capture the filenames under directory
        files = glob.glob(os.path.join(args.video_location, '*.mp4'))
        for fileName in files:
            #Run pipeline for the video
            file_output = 'test_results/0'+ os.path.basename(fileName)
            print("Processing video", os.path.basename(fileName))
            clip1 = VideoFileClip(args.video_location+os.path.basename(fileName))
            white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
            white_clip.write_videofile(file_output, audio=False)
            print("Video saved successfully")

    print("Program completed :)")
