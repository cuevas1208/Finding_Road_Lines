## File: drawLineFunctions.py
## Name: Manuel Cuevas
## Date: 12/08/2016
## Project: CarND - LaneLines
## Usage: This file containes a series of fucntions that can help identify and draw the
## road lane lines from images and video.
## Tools learned on the CarND program were used on this file
## Revision: Rev 0000.004 12_11_2016
#######################################################################################
#importing useful packages
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""colorTransformation
Mask an image white and yellow colors.
Input: img, image 8-bit, 16-bit unsigned or single-precision floating-point
Return: A color filter for a scpectrum of white and yellow colors
        The spectrum of white can be modefied by changeing color_thresholds
        The spectrum of yellow can be modify by changing the lower_color and upper_color
"""
def colorTransformation(image):
    ###################COLOR SELECT - WHITE line #####################
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

    #Uncomment the following code if you wish to save the image
    #mpimg.imsave("test_results/01colorSelection01.png", gray)
    #print ('colorSelection completed')
    
    ###################COLOR RANGE - YELLLOW line ####################
    # define range of blue color in HSV
    lower_color = np.array([200,190,0])
    upper_color = np.array([255,255,160])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image, lower_color, upper_color)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)

    # merger the grey image and the new image
    color_filter = cv2.bitwise_or(gray, res)
    
    return color_filter

"""maskImage
Triangle mask with dimentions left_bottom, right_bottom, and apex
Input: img, image 8-bit, 16-bit unsigned or single-precision floating-point
Return: An image with a triangle mask
"""
def maskImage(image, mask_filter):
    ###################MASK LINES#####################################
    # Define a triangle region of interest
    ysize = image.shape[0]
    xsize = image.shape[1]
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

    # Find where the image is both colored right and in the region
    mask_filter[~region_thresholds] = [0,0,0]
    return mask_filter

"""region_of_interest
Only keeps the region of the image defined by the polygon
formed from `vertices`. The rest of the image is set to black.
Input: img - image 8-bit, 16-bit unsigned or single-precision floating-point
       threshold1 – first threshold for the hysteresis procedure
       threshold2 – second threshold for the hysteresis procedure
Return: An image with the mask edges (https://en.wikipedia.org/wiki/Canny_edge_detector)
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

right_Ypair, right_Xpair = None, None
left_pair, left_x = None, None

"""draw_lines
In two ways by drawing the lines already detected and by using these lines to  
decide which segments are part of the left line vs the right lane by the  
slope ((y2-y1)/(x2-x1)) and extrapolate the line segments.
Input: img - image 8-bit, 16-bit unsigned or single-precision floating-point
       lines - array of lines arleady detected
       color - color to draw lines
       thickness - the thickness of the lines to be drawn
Returns: An image with the road lines drawn
"""
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    right_slope = []
    left_slope  = []
    left_lines  = []
    right_lines = []
    
    global right_Ypair, right_Xpair, left_Ypair, left_Xpair
        
    for line in lines:
    
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
            m = ((y2-y1)/(x2-x1)) # slope
            if m < 0:
                left_slope.append(m)
                left_lines.append((x2,y2))
            else:
                right_slope.append(m)
                right_lines.append((x1,y1))
    thickness=5

    try:
        if (left_slope):
            left_slope = sorted(left_slope)[int(len(left_slope)/2)]
            #get Y location
            left_y1 = min([line[1] for line in left_lines])
            left_Ypair = tuple([line[0] for line in left_lines if line[1] == left_y1] + [left_y1])
            
            #get x location
            left_x = int((img.shape[1]-left_Ypair[1])/left_slope) + left_Ypair[0]
            left_Xpair = (left_x, img.shape[1])
            
        cv2.line(img, left_Ypair, left_Xpair, color, thickness)
    except:
        pass
    
    
    try:
        if (right_slope):
            right_slope = sorted(right_slope)[int(len(right_slope)/2)]
            #get Y location
            right_y1 = min([line[1] for line in right_lines])
            right_Ypair = tuple([line[0] for line in right_lines if line[1] == right_y1] + [right_y1])

            #get x location
            right_x = int((img.shape[1]-right_Ypair[1])/right_slope) + right_Ypair[0]
            right_Xpair = (right_x, img.shape[1])
            
        cv2.line(img, right_Ypair, right_Xpair, color, thickness)
    except:
        pass
    
    
    return

"""hough_lines
Returns an image with hough lines drawn.  
Input: img - image 8-bit, 16-bit unsigned or single-precision floating-point
       rho - array of lines arleady detected
       theta - color to draw lines
       threshold - the thickness of the lines to be drawn
       min_line_len
       max_line_gap
Returns: A back image with the road lines drawn
"""
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

"""printImg
Plots two images side by side for comparing    
"""
def printImg(img1, img2, img1_title = 'Input Image', img2_title = 'Output Image', cmap='Greys_r'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1, cmap)
    ax1.set_title(img1_title, fontsize=50)
    ax2.imshow(img2)
    ax2.set_title(img2_title, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
