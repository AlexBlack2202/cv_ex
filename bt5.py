import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('h3.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



# chọn ngưỡng
# def nothing(x):
#     pass

# cv2.namedWindow('canny')
# cv2.createTrackbar('lower', 'canny', 0, 255, nothing)
# cv2.createTrackbar('upper', 'canny', 0, 255, nothing)
# while(1):
#     lower = cv2.getTrackbarPos('lower', 'canny')
#     upper = cv2.getTrackbarPos('upper', 'canny')

#     ret, thresh = cv2.threshold(gray,lower,upper,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#     # display images
#     #cv2.imshow('original', img)
#     cv2.imshow('canny', thresh)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:   # hit escape to quit
#         break

# cv2.destroyAllWindows()




ret, thresh = cv2.threshold(gray,10,55,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.imshow(thresh,cmap = 'gray')
plt.title("thresh")
plt.show()

# noise removal
kernel = np.ones((1,1),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

plt.imshow(opening,cmap = 'gray')
plt.title("noise removal")
plt.show()

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

plt.imshow(unknown,cmap = 'gray')
plt.title("unknown")
plt.show()


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

plt.imshow(markers,cmap = 'gray')
plt.title("markers")
plt.show()

# Now, mark the region of unknown with zero
markers[unknown>2] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.imshow(img,cmap = 'gray')
plt.title("img Watershed")
plt.show()