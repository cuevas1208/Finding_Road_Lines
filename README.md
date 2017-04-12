# Project: Finding Lane Lines on the Road

Overview
---
The project uses a pipeline to detect the road line segments, then extrapolate them and draw them onto the image for display. Within the pipeline some of the tools used are: color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Transform line detection.

Quickstart
---
Download repository
To run the program on the existing images under "test_images". Open command line under the project directory and type the following.
```sh
python main.py
```
Note: You can replace the images found in "test_images" with your own road images

(Optional) If you would like to process videos, all you have to do is add to the command line an argument with the directory where the videos are located.Example:
```sh
python main.py ../test_video/
```
The output video or images will be stored under test_results directory

Dependencies
---
Python 3.5, openCV 3.2

References
---
This project is part of the CarND program. Tools learned in class were used to identify lane lines on the road.
