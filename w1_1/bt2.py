import cv2

# read image
img = cv2.imread('h1.png')
#show image
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#convert image to grayscale and save file
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_img.jpg',gray_image)

#reload gray image and display

gray_img = cv2.imread('gray_img.jpg',cv2.IMREAD_GRAYSCALE)


cv2.imshow('img_gray',gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()