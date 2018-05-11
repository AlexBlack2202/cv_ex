import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('h1.jpg',0)

# edges = cv2.Canny(img,100,210)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()

height,width = img.shape

# set max width = 800 px

if width > 800:
    ratio = width / 800
    new_height = int(height/ratio)

    img = cv2.resize(img, (800, new_height)) 


def nothing(x):
    pass

# create trackbar for canny edge detection threshold changes
cv2.namedWindow('canny - press esc for exit')

# add lower and upper threshold slidebars to "canny"
cv2.createTrackbar('lower', 'canny', 0, 255, nothing)
cv2.createTrackbar('upper', 'canny', 0, 255, nothing)

# Infinite loop until we hit the escape key on keyboard
while(1):

    # get current positions of four trackbars
    lower = cv2.getTrackbarPos('lower', 'canny')
    upper = cv2.getTrackbarPos('upper', 'canny')
    
    edges = cv2.Canny(img, lower, upper)

    # display images
    #cv2.imshow('original', img)
    cv2.imshow('canny', edges)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:   # hit escape to quit
        break

cv2.destroyAllWindows()


