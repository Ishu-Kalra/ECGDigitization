import cv2
import numpy as np
import random

def invert(im):
	im2=255-im
	return im2
'''
def maskLength(im):
	mask = 0
	rows, cols = im.shape[0:2] #Unpacking Rows and Columns
	selection_range = cols // 5 #Dividing image in regions of 5 pixel length
	random_regions = []
	for i in range(5):
		random_regions.append(random.randint(0, len(selection_range)-1)) #Selecting randomly 5 regions
	for region in random_regions:
		start_pixel = region * 5
		for row in rows:
			c = 0
			for j in range(5):
				if im[row][j + start_pixel] == 255:
					c += 1
				if im[row][j + start_pixel] != 255:
					break
			mask = max(c, mask)  #Maximum continous pixel length in selected regions
	return mask #Length of mask
	'''


def maskApplication(im):
	mask = 5 #Can later calculate the very length of mask
	row, col = im.shape

	for i in range(row // mask):
		for j in range(col // mask):
			pass
'''
---------------------
|p23|p24|p9	|p10|p11|
---------------------
|p22|p8	|p1	|p2	|p12|
---------------------
|p21|p7	|px	|p3	|p13|
---------------------
|p20|p6	|p5	|p4	|p14|
---------------------
|p19|p18|p17|p16|p15|
---------------------
The spiral 5X5 mask with the central pixel as px

Rules are applied as follows

'''

#Commenting out the rules for now to check CV Build
'''
def ruleOne(im, mask, rowSet, columnSet):
	centralPixel = [rowSet*mask+math.ceil(mask / 2), columnSet*mask+math.ceil(mask / 2)]
	isRuleTrueForOR = False
	isRuleTrueForAND = True
	#Performing OR rule first and AND operation later for logics' sake
	for i in range(mask):
		if (im[rowset*mask+1][columnSet*mask+i] == 0):
			isRuleTrueForAND = False
	 
	 for j in range(3):
		for i in range(mask):
			if (im[rowSet*mask+2+j][columnSet*mask+i] != 255):
				isRuleTrueForOR = True
				break
	#im[centralPixel[0], centralPixel[1]] = 255
	if (isRuleTrueForOR and isRuleTrueForAND):
		im[centralPixel[0]][centralPixel[1]] = 0
		
def ruleTwo(im, mask, rowSet, columnSet):
'''
	#White pixel for AND: i = 1 to 5, 10 and 16
	#Black Pixel for OR: j = 6, 7, 8, 18, 24
'''
	centralPixel = [rowSet*mask+math.ceil(mask / 2), columnSet*mask+math.ceil(mask / 2)]
	isRuleTrueForOR = False
	isRuleTrueForAND = True
	#Performing OR rule first and AND operation later for logics' sake
	for i in range(mask):
		if (im[rowset*mask+i][columnSet*mask+1] == 0):
			isRuleTrueOR = True
	for i in range(mask):
		if (im[rowSet*mask+i][columnSet*mask+3] != 255):
			isRuleTrueForAND = False
			break
	#im[centralPixel[0], centralPixel[1]] = 255
	for i in range(1, mask-1):
		if (im[rowSet*mask+i][columnSet*mask+2] != 255):
			isRuleTrueforAND = False
			break
	if (isRuleTrueForOR and isRuleTrueForAND):
		im[centralPixel[0]][centralPixel[1]] = 255
		
def ruleThree(im, mask, rowSet, columnSet):
	centralPixel = [rowSet*mask+math.ceil(mask / 2), columnSet*mask+math.ceil(mask / 2)]
	isRuleTrueForOR = False
	isRuleTrueForAND = True
	#Performing OR rule first and AND operation later for logics' sake
	for i in range(mask):
		if (im[rowset*mask+i][columnSet*mask+3] == 0):
			isRuleTrueOR = True
	for i in range(mask):
		if (im[rowSet*mask+i][columnSet*mask+1] != 255):
			isRuleTrueForAND = False
			break
	#im[centralPixel[0], centralPixel[1]] = 255
	for i in range(1, mask-1):
		if (im[rowSet*mask+i][columnSet*mask+2] != 255):
			isRuleTrueforAND = False
			break
	if (isRuleTrueForOR and isRuleTrueForAND):
		im[centralPixel[0]][centralPixel[1]] = 255

def ruleFour(im, mask, rowSet, columnSet):
	centralPixel = [rowSet*mask+math.ceil(mask / 2), columnSet*mask+math.ceil(mask / 2)]
	isRuleTrueForOR = False
	isRuleTrueForAND = True
	isRuleTrueForORSecond = False
		#Performing OR rule first and AND operation later for logics' sake
	for i in range(2, mask):
		if (im[rowset*mask+3][columnSet*mask+i] == 0):
			isRuleTrueOR = True
	for i in range(0, 2):
		if (im[rowSet*mask+3][columnSet*5+i] == 255):
			isRuleTrueForORSecond ==  True
	for i in range(mask):
		if (im[rowSet*mask+1][columnSet*mask+i] != 255):
			isRuleTrueForAND = False
			break
				#im[centralPixel[0], centralPixel[1]] = 255
	for i in range(1, mask-1):
		if (im[rowSet*mask+2][columnSet*mask+i] != 255):
			isRuleTrueforAND = False
			break
	if (isRuleTrueForOR and isRuleTrueForAND and isRuleTrueForORSecond):
		im[centralPixel[0]][centralPixel[1]] = 255

def ruleFive(im, mask, rowSet, columnSet):
	centralPixel = [rowSet*mask+math.ceil(mask / 2), columnSet*mask+math.ceil(mask / 2)]
	isRuleTrueForOR = False
	isRuleTrueForAND = True
	isRuleTrueForORSecond = False
		#Performing OR rule first and AND operation later for logics' sake
	for i in range(2, mask-2):
		if (im[rowset*mask+3][columnSet*mask+i] == 0):
			isRuleTrueOR = True
	for i in range(0, 2):
		if (im[rowSet*mask+3][columnSet*5+i] == 255):
			isRuleTrueForORSecond ==  True
	for i in range(mask):
		if (im[rowSet*mask+1][columnSet*mask+i] != 255):
			isRuleTrueForAND = False
			break
	#im[centralPixel[0], centralPixel[1]] = 255
	for i in range(1, mask-1):
		if (im[rowSet*mask+2][columnSet*mask+i] != 255):
			isRuleTrueforAND = False
			break
	if (isRuleTrueForOR and isRuleTrueForAND and isRuleTrueForORSecond):
		im[centralPixel[0]][centralPixel[1]] = 255

def ruleSix(im, mask, rowSet, columnSet):
	centralPixel = [rowSet*mask+math.ceil(mask / 2), columnSet*mask+math.ceil(mask / 2)]
	isRuleTrueForOR = False
	isRuleTrueForAND = True
	#isRuleTrueForORSecond = False
		#Performing OR rule first and AND operation later for logics' sake
	for i in range(mask):
		if (im[rowset*mask+1][columnSet*mask+i] == 0):
			isRuleTrueOR = True
	#for i in range(0, 2):
	#	if (im[rowSet*mask+3][columnSet*5+i] == 255):
	#		isRuleTrueForORSecond ==  True
	for i in range(mask):
		if (im[rowSet*mask+3][columnSet*mask+i] != 255):
			isRuleTrueForAND = False
			break
	#im[centralPixel[0], centralPixel[1]] = 255
	for i in range(1, mask-1):
		if (im[rowSet*mask+2][columnSet*mask+i] != 255):
			isRuleTrueforAND = False
			break
	if (isRuleTrueForOR and isRuleTrueForAND):
		im[centralPixel[0]][centralPixel[1]] = 255
'''

def thinningPartOne(im):
	row, col = im.shape
	distanceArray = x = [[0 for i in range(row)] for j in range(col)] #Default distance is 0
	for j in range(col):
		for i in range(row):
			pixelProcessed = [i, j]
			for down in range(i, row):
				if im[down][j] == 255:
					distanceArray[i][j] = math.abs(down - i)
					break
	
	return distanceArray

def postFiltering(im):
	pass

def textDetection(im):
	
	rgb = cv2.pyrDown(im)
	small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

	_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
	connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
	# using RETR_EXTERNAL instead of RETR_CCOMP
	print(cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE))

	mask = np.zeros(bw.shape, dtype=np.uint8)

	for idx in range(len(contours)):
		x, y, w, h = cv2.boundingRect(contours[idx])
		mask[y:y+h, x:x+w] = 0
		cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
		r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

		if r > 0.45 and w > 8 and h > 8:
			cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

	cv2.imshow('rects', rgb)

	

im = cv2.imread("ECGNormal.png")
for i in range(im.shape[0]):
	for j in range(im.shape[1]):
		if im[i][j][0] < 70 and im[i][j][1] < 70 and im[i][j][2] < 70:
			im[i][j] = 0
		else:
			im[i][j] = 255
cv2.imshow("process step1",im)
cv2.imwrite("./output_no_inversion_no_dilation.jpg",im)
#kernel = np.ones((3,3), np.uint8)
kernel_erosion = np.ones((10, 10), np.uint8)
#for i in range(5):
im3 = cv2.erode(im, kernel_erosion, iterations = 1)
#im3 = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('./output_no_inversion_EROSION.jpg', im3)	
#cv2.imwrite('./output_no_inversion_no_dilation_morphological_close.jpg', im3)
kernel = np.ones((3,3),np.uint8)
im2=cv2.dilate(im,kernel,iterations=1)
cv2.imwrite("./output_no_inversion_DILATION.jpg",im2)
im2=invert(im2)

#cv2.imshow("process step2", im2)

#cv2.imshow("process step3",im2)

cv2.imwrite("./output.jpg",im2)

textDetection(im)

