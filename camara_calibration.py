## File: camara_calibration.py
## Name: Manuel Cuevas
## Date: 3/08/2016
## Project: CarND - LaneLines
## Desc: Camera Calibration with OpenCV
## Usage: This class calibrates the camara by using a so-called pinhole camera model
## http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.findChessboardCorners
#######################################################################################
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

class imgCalib():
    ''' calibrate
    Extracts object points and image points for camera calibration.
    Input:  imgLoc - Directory where the chest images are located
    Output: outputFileName - File name, where the calibration parameters
            will be saved  
    '''
    def calibrate(imgLoc = 'test_images/calibrationImages', outputFileName = "wide_dist_pickle.p"):
        #Count the number of corners in any given row and enter that value in nx.
        #Similarly, count the number of corners in a given column and store that in ny
        nx = 9 # the number of inside corners in x
        ny = 6 # the number of inside corners in y

        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(imgLoc + '/*.jpg')

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                #Draw the corners and save them unde cornersFound
                import os.path
                if not(os.path.isdir(imgLoc+"/cornersFound")):
                    os.makedirs(imgLoc+"/cornersFound")
                cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                write_name = imgLoc +'/cornersFound/corners_found'+str(idx+1)+'.jpg'
                cv2.imwrite(write_name, img)
            
        # Test undistortion on an image
        img_size = (img.shape[1], img.shape[0])
        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open( imgLoc+"/"+outputFileName, "wb" ) )

    ''' showCalibrate 
    Compares the undistorted images to the original images 
    Input:  imgLoc - Directory where the chest images are located
    '''
    def showCalibrate(imgLoc = 'test_images/calibrationImages'):
        import numpy as np
        import cv2
        import glob
        import matplotlib.pyplot as plt
        import pickle
        get_ipython().magic('matplotlib inline')

        # Load the camera calibration result 
        dist_pickle = pickle.load(open( imgLoc + "/wide_dist_pickle.p", "rb" ) )
        mtx = dist_pickle['mtx']
        dist = dist_pickle['dist']

        img = cv2.imread( imgLoc + '/calibration1.jpg')
        img_size = (img.shape[1], img.shape[0])

        undist = cv2.undistort(img, mtx, dist, None, mtx)

        #save image
        cv2.imwrite(imgLoc+'/test_undist.jpg',undist)

        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=30)

    ''' PerspectiveTransform 
    Creates a bird’s eye view representation of an image
    Some tasks are easier to perform in a bird’s eye image like
    finding the curvature of the lane
    Input:  img - image to be process
    Output: warp - perspective transform image
            M - perspective transform matrix
    '''
    def PerspectiveTransform(img, plotOutput = 'No'):
        # Test undistortion on an image
        img_size = (img.shape[1], img.shape[0])

        # For source points I'm grabbing the four detected points
        src = np.float32([[1115, 690], [845, 530], [280, 690], [500, 530]])

        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[1115, 690], [1115, 530], [280, 690], [280, 530]])

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warp = cv2.warpPerspective(img, M, img_size)

        if plotOutput == 'Yes':
            # Plot the result
            # printImg(img, warp, img2_title = 'Perspective Transform Image')
            mpimg.imsave("output_images/PerspectiveTransform.png", warp)

        return (warp, M)


