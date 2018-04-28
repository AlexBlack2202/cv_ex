
#bai tap 5
import cv2
from matplotlib import pyplot as plt
# read color image and resize
img = cv2.imread('h1.png',cv2.IMREAD_COLOR )




def gauss_filter(img, size,sigma=0):
    return cv2.GaussianBlur(img,size,sigma)



fig1, ax1 = plt.subplots(nrows=5, ncols=2)
ax1[0, 0].imshow(gauss_filter(img,(11,11),3))
ax1[0, 0].get_xaxis().set_ticks([])
ax1[0, 0].get_yaxis().set_ticks([])
ax1[0, 0].set_title("11-3")

ax1[0, 1].imshow(gauss_filter(img,(11,11),9))
ax1[0, 1].get_xaxis().set_ticks([])
ax1[0, 1].get_yaxis().set_ticks([])
ax1[0, 1].set_title("11-9")

ax1[1, 0].imshow(gauss_filter(img,(51,51),3))
ax1[1, 0].get_xaxis().set_ticks([])
ax1[1, 0].get_yaxis().set_ticks([])
ax1[1, 0].set_title("51-3")

ax1[1, 1].imshow(gauss_filter(img,(51,51),9))
ax1[1, 1].get_xaxis().set_ticks([])
ax1[1, 1].get_yaxis().set_ticks([])
ax1[1, 1].set_title("51-9")

ax1[2, 0].imshow(gauss_filter(img,(101,101),3))
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 0].set_title("101-3")

ax1[2, 1].imshow(gauss_filter(img,(101,101),9))
ax1[2, 1].get_xaxis().set_ticks([])
ax1[2, 1].get_yaxis().set_ticks([])
ax1[2, 1].set_title("101-9")

ax1[3, 0].imshow(gauss_filter(img,(3,3),50))
ax1[3, 0].get_xaxis().set_ticks([])
ax1[3, 0].get_yaxis().set_ticks([])
ax1[3, 0].set_title("3-50")

ax1[3, 1].imshow(gauss_filter(img,(201,201),20))
ax1[3, 1].get_xaxis().set_ticks([])
ax1[3, 1].get_yaxis().set_ticks([])
ax1[3, 1].set_title("201-20")

ax1[4, 0].imshow(img)
ax1[4, 0].get_xaxis().set_ticks([])
ax1[4, 0].get_yaxis().set_ticks([])
ax1[4, 0].set_title("anh goc")

ax1[4, 1].imshow(img)
ax1[4, 1].get_xaxis().set_ticks([])
ax1[4, 1].get_yaxis().set_ticks([])
ax1[4, 1].set_title("anh goc")
plt.show()
# Kết quả cho ta thấy
# khi size và sigma nhỏ: ảnh không thay đổi nhiều so với ảnh ban đầu (hình 1 bên trái)
# khi size nhỏ và sigma lớn: ảnh sắc nét hơn ( hình số 4 ở bên trái)
# khi size lớn và sigma nhỏ: ảnh cũng không thay đổi nhiều so với ảnh ban đầu (hình số 3 bên trái)
# khi size và sigma đều lớn: ảnh trông bị nhoè đi


# bai tap 6

#khi ta điều chỉnh tiêu cự của ống kính máy ảnh thì x và y của ảnh thay đổi, z không thay đổi, trong trường hợp giảm tiêu cự thì x và y tăng lên, làm  cho hình ảnh cân đối hơn
#còn đối với trường hợp tăng tiêu cự thì bức ảnh trông sẽ tệ hơn