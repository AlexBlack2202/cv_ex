import cv2

# read color image and resize
img = cv2.imread('h1.png',cv2.IMREAD_COLOR )

img_resize = cv2.resize(img,(256,256))

cv2.imshow('img',img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('img_resize.jpg',img_resize)


# read grayscale image and resize
img_gray = cv2.imread('gray_img.jpg' )

img_gray_resize = cv2.resize(img_gray,(256,256))

cv2.imshow('img',img_gray_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('img_gray_resize_resize.jpg',img_gray_resize)