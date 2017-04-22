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
from advPipeline import advLine
from basicPipeline import basicLine
print(sys.version)

''' main
Saves the pipeline output in to test_results directory
Input: Directory location of the videos to process
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Arguments')
    parser.add_argument(
        'pipeline',
        nargs='?',
        type=int,
        default=2,
        help='Basic pipeline = 1, Advance pipleline = 2'
    )
    parser.add_argument(
        'video_location',
        type=str,
        nargs='?',
        default='',
        help='Path to video folder. This is where the video will be loaded from'
    )
    args = parser.parse_args()
    if (args.pipeline == 1):
        Pipeline = basicLine()
        outPutLocation = "test_results/basic"
    else:
        Pipeline = advLine()
        outPutLocation = "test_results/advance"
        
    #Create output directory 
    if not os.path.exists(outPutLocation):
        os.makedirs(outPutLocation)

    ### Test on Videos
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip

    if args.video_location != '':
        #capture the filenames under directory
        files = glob.glob(os.path.join(args.video_location, '*.mp4'))
        for fileName in files:
            #Run pipeline for the video
            file_output = outPutLocation+'/V_'+ os.path.basename(fileName)
            print("Processing video", os.path.basename(fileName))
            clip1 = VideoFileClip(args.video_location+os.path.basename(fileName))
            white_clip = clip1.fl_image(Pipeline.pipeline) #NOTE: this function expects RGB images!!
            white_clip.write_videofile(file_output, audio=False)
            print("Video saved successfully")
    else:
        ### Test on Images
        # Build pipeline to work with the images in the directory "test_images/"
        if os.path.exists("test_images/"):
            #imagesList = os.listdir("test_images/*.jpg")
            imagesList = glob.glob(os.path.join("test_images/", '*.jpg'))
            print("Processing ", len(imagesList), " images...")
            for i, imageLocation in (enumerate(imagesList)):
                #open file
                image = mpimg.imread(imageLocation)
                result = Pipeline.pipeline(image)
                
                # Save & Display the results
                mpimg.imsave(outPutLocation+"/R_"+os.path.basename(imageLocation)+".png", result)
                plt.figure()
                plt.imshow(result)
                print("Images", i, "saved successfully")
        else:
            print("Missing images dirctory test_images")


    print("###Program completed###")
