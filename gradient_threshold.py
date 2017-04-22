import numpy as np
import cv2

class grad_threshold():
    def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # Apply threshold
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        if orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        abs_sobel = np.absolute(sobel) 
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # Create a mask of 1's where the scaled gradient magnitude 
        # is > thresh_min and < thresh_max
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return grad_binary

    def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        # Calculate the magnitude 
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scaled_sobel = np.uint8(255*gradmag/np.max(gradmag))

        thresh_min = 20
        thresh_max = 100
        mag_binary = np.zeros_like(scaled_sobel)
        mag_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return mag_binary

    def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # Apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        absgraddir = np.arctan2(abs_sobely, abs_sobelx)

        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        return dir_binary

    def color_Threshold(img, v_thresh=(0, 255), s_thresh=(0, 255)):
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        v_channel = hsv[:,:,2]

        # Threshold color channel
        v_binary = np.zeros_like(v_channel)
        v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        s_channel = hls[:,:,2]

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        combined = np.zeros_like(s_binary)
        combined[((s_binary == 1) & (v_binary == 1))] = 1

        #color_binary = np.dstack(( np.zeros_like(v_binary), v_binary, s_binary))

        return combined