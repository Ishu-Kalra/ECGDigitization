import cv2
import numpy as np

def invert(im):
	im2=255-im
	return im2

im = cv2.imread("ECGNormal.png")
print(im.shape)


for i in range(im.shape[0]):
	for j in range(im.shape[1]):
		if im[i][j][0] < 70 and im[i][j][1] < 70 and im[i][j][2] < 70:
			im[i][j] = 255
		else:
			im[i][j] = 0
#cv2.imshow("process step1",im)
cv2.imwrite("./output_no_inversion_no_dilation.jpg",im)
kernel = np.ones((3,3),np.uint8)
im2=cv2.dilate(im,kernel,iterations=1)
cv2.imwrite("./output_no_inversion.jpg",im2)
im2=invert(im2)

#cv2.imshow("process step2", im2)

#cv2.imshow("process step3",im2)

cv2.imwrite("./output.jpg",im2)
