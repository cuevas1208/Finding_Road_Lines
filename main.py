## File: main.py
## Name: Manuel Cuevas
## Date: 12/03/2016
## Project: CarND - LaneLines
## Desc: Pipeline uses CV techniques like image threshold, GaussianBlur, Canny,
## and HoughLinesP. 
## Usage: This project identifies a series of steps that identify and draw the
## road lane lines on a from images and video.

#import some useful package
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#open file
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##################################################################
###################COLOR SELECT###################################

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
gray = np.copy(image)

# Define color selection criteria
red_threshold = 180
green_threshold = 180
blue_threshold = 180

rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Do a bitwise or with the "|" character to identify
# pixels below the thresholds
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])
gray[color_thresholds] = [0,0,0]

# Display the image                 
# cv2.imshow("color",gray)

# Uncomment the following code if you are running the code locally and wish to save the image
mpimg.imsave("test_images/01colorSelection01.png", gray)
print ('colorSelection completed')
##################################################################
###################COLOR RANAGE SELCTION##########################
#Ref http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
'''
# Convert BGR to HSV
#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# define range of blue color in HSV
lower_blue = np.array([220,220,220])
upper_blue = np.array([250,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(image, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(image,image, mask= mask)

cv2.imshow('renge',res)

mpimg.imsave("test_images/02renge02.jpg", res)
print ('02res02 completed')
'''
###################FillPoly#######################################
#for regions selection                                           #
##################################################################
'''
left_top = [xsize/2-5, ysize*(1/2)+20]
right_top = [xsize/2+5, ysize*(1/2)+20]
right_bottom = [xsize-35, ysize]
left_bottom = [35, ysize]
a3 = np.array( [[left_top,right_top,right_bottom,left_bottom]], dtype=np.int32 )
im = np.zeros([240,320],dtype=np.uint8)
cv2.fillPoly( image, a3, 255 )

plt.imshow(image)
plt.show()
'''
##################################################################
###################MASK LINES#####################################
# Grab the x and y sizes and make two copies of the image
# With one copy we'll extract only the pixels that meet our selection,
# then we'll paint those pixels red in the original image to see our selection 
# overlaid on the original.
line_image = np.copy(image)

# Define a triangle region of interest (Note: if you run this code, 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
# you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz ;)
left_bottom = [35, ysize]
right_bottom = [xsize-35, ysize]
apex = [xsize/2, ysize*(1/2)]
print(int(ysize*(1/15)))
# Fit lines (y=Ax+B) to identify the  3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255,0,0]
gray[~region_thresholds] = [0,0,0]

# Display our two output images
#plt.imshow(color_select)
#plt.imshow(line_image)

mpimg.imsave("test_images/region.png", gray)
mpimg.imsave("test_images/02Mask01.png", line_image)
print('Mask Completed')
##################################################################
###################Gaussian#######################################
# Define a kernel size and apply Gaussian smoothing
kernel_size = 7
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)


# Define our parameters for Canny and apply
low_threshold = 50      #wiet, make line think
high_threshold = 300     #eleminate the background
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap='Greys_r')
mpimg.imsave("test_images/03Gaussian01.png", edges, cmap='Greys_r')
print ('Gaussian completed')
##################################################################
###################Hough transform################################
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 120
max_line_gap = 80
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
        
# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges))

# Draw the lines on the edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
#cv2.imshow('04solidWhiteRight_pipe01.png',combo)

mpimg.imsave("test_images/04solidWhiteRight_pipe01.png", combo)
print ('Hough completed')

##################################################################
###################weighted_img################################
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

combo = weighted_img(combo, image)
# Display the image
#cv2.imshow('04solidWhiteRight_pipe01.png',combo)
