# openCV uses channel BGR instread of RGB
import numpy as np
import pandas as pd
import cv2

image = cv2.imread('Lane.png')

# convert the RGB image to Gray scale
grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bc there is so many lines detected here, we only need lines of the land =>
# Apply a mask polygon to only focus on the road using region of interest

# create an array of the same size as of the input image 
# Ex: [[1,2], [3,4]] => [[0,0], [0,0]]
mask = np.zeros_like(image)

# print(image.shape ) # HEIGHT X WIDTH

# create a white mask (255) corresponding to the number of channels of the picture
if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count  # print((3,)* 3) <=> (3, 3, 3)
else:
        ignore_mask_color = 255

# create a polygon to surround the road
rows=  image.shape[0]
cols= image.shape[1]

# 4 points/vertices of the polygon
# remember the Oxy axes here has the Oy upsidedown 
bottom_left  = [cols * 0.09, rows * 1]
top_left     = [cols * 0.55, rows * 0.56]
bottom_right = [cols * 0.92, rows * 1]
top_right    = [cols * 0.57, rows * 0.56]

vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32) #data type is integer 32bit

# filling the polygon white mask
cv2.fillPoly(mask, vertices, ignore_mask_color)

# performing Bitwise AND on the input image and mask to get the binary image of the road
# which mask bit is 0 will be set to 0, where mask bit is 1, the corresponding bit of the image will retain 
masked_image = cv2.bitwise_and(image, mask)

# blur to reduce noise for the canny detection
# and focuses on our region of interest
# size of gaussian kernel
kernel_size = 7
blur = cv2.GaussianBlur(masked_image, (kernel_size, kernel_size), 0)
# cv2.imshow('Lane Detecting', blur )

# applying canny edge detection and save edges in a variable
CannyEdge = cv2.Canny(blur, 50, 150)
# cv2.imshow('Lane Detecting', CannyEdge)

# Probabilistic Hough Transform => to get straight line 
        # Distance resolution of the p(ro) in pixels.
p = 1             
    # Angle resolution of the accumulator in radians.
theta = np.pi/180   
    # minimum votes to be considered a line
threshold = 20      
    # Line segments shorter than that are rejected.
minLineLength = 20  
    # Maximum allowed gap between points on the same line to link them
# maxLineGap = 500    
    # An array containing dimensions of straight lines 
#     NOTE: THE IMAGE HERE HAS TO BE THE RESULT OF CANNY DETECTING STEPS
HoughLines=cv2.HoughLinesP(CannyEdge, p, theta, threshold)

print(HoughLines.shape)

# Draw lines
line_image = np.zeros_like(image)
if HoughLines is not None:
    for line in HoughLines:
        x1,y1, x2,y2 = line[0]
        cv2.line(line_image,(x1,y1), (x2,y2), (0,255,0), 5)

final=cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
cv2.imshow('Lane Detecting', line_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()