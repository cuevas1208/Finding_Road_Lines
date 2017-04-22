## File: advPipeline.py
## Name: Manuel Cuevas
## Date: 3/08/2016
## Desc: Pipeline uses CV techniques to implement camera calibration and transforms,
#         as well as filters, polynomial fits, and splines.
## Usage: Detect lane lines in a variety of conditions, including changing road
#         surfaces, curved roads, and variable lighting. 
#######################################################################################
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import os.path
from drawLineFunctions_adv import advLaneFind
from camara_calibration import imgCalib
from gradient_threshold import grad_threshold

class advLine:
    ''' cameraCalibration
    Extracts object points and image points for camera calibration.
    Input:  img - RGB image of the road
     plotOutput - yes if you like to save the image
    Output: result - RGB image of the road, with the road lines drawn  
    '''
    def cameraCalibration(self, img, plotOutput = 'No'):
        fileName = "wide_dist_pickle2.p"
        imgLoc = "test_images/calibrationImages"
        calibratedImages = imgLoc+"/"+fileName
        if not os.path.isfile(calibratedImages):
            #Calibrate Images
            imgCalib.calibrate(imgLoc, outputFileName = fileName)
         
        dist_pickle = pickle.load(open( calibratedImages, "rb" ) )
        mtx = dist_pickle['mtx']
        dist = dist_pickle['dist']
        undist = cv2.undistort(img, mtx, dist, None, mtx) 

        # Plot the result
        if plotOutput == 'Yes':
            mpimg.imsave("output_images/undistortedImage.png", undist)
        return undist

    ''' gradThreshold
    Color & Gradient Thresholding functions filter the color image, excluding part of the image
    that are irrelevant to the image processing 
    Input:  img - RGB image of the road
    Output: result - RGB image of the road, with the road lines drawn  
    '''
    def gradThreshold(self, img, plotOutput = 'No'):
        
        ksize = 3 # Choose a larger odd number to smooth gradient measurements

        gradx = grad_threshold.abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(12, 255))
        grady = grad_threshold.abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(25, 255))
        c_Threshold = grad_threshold.color_Threshold(img, v_thresh=(115, 255), s_thresh=(115, 255))
        
        combined = np.zeros_like(img[:,:,0])
        combined[((gradx == 1) & (grady == 1))|(c_Threshold == 1)] = 255

        if plotOutput == 'Yes':
            # Plot the result
            mpimg.imsave("output_images/gradThreshold.png", combined)
        return combined


    ''' pipeline
    Draw road line Pipeline 
    Input:  img - RGB image of the road
    Output: result - RGB image of the road, with the road lines drawn  
    '''
    def pipeline(self, img):
        #Camera Calibration
        img_undist = self.cameraCalibration(img, plotOutput = 'No')
        #Color & Gradient threshold
        img_grand = self.gradThreshold(img_undist, plotOutput = 'No')
        #Perspective Transform
        img_PT, M = imgCalib.PerspectiveTransform(img_grand, plotOutput = 'No')  
        #Locate the Lane Lines and Fit a Polynomial
        AdvLaneFind = advLaneFind()
        AdvLaneFind.fitPoly(img_PT, plotOutput = 'No')
        #Measuring Curvature
        img_undist, left_fitx, right_fitx, ploty = AdvLaneFind.laneCurvature(img_PT, img_undist, plotOutput = 'No')
        #Draw image lines
        result = AdvLaneFind.drawImgLines(img, img_undist, img_PT, M, left_fitx, right_fitx, ploty, plotOutput = 'No')  
        
        return result

