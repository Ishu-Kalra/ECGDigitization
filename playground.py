import cv2
import numpy as np
from PIL import Image
from PIL import EpsImagePlugin
import pytesseract
import math
import re
import time

gray = cv2.imread('AfterSigThresholding.png')
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) #necessary to always convert when you read from imread. IMREAd automatically makes it 3 channels
arr = np.array([1,2,3,4,5,6,7,8,9])
print(np.flipud(arr))
start = time.time()
maxNoPeakLength = 30
kernel = np.ones((3,3),np.uint8)
gray=cv2.dilate(cv2.bitwise_not(gray),kernel,iterations=1)
gray = cv2.bitwise_not(gray) #Dilating to join the whole signal. Otherwise some column will exist without any black pixels
for i in range(gray.shape[1]):
    Xup = None
    Xdn = None #Will take negative value of this as pixels are flipped here
    #Xdn is negative but referes to to the down edge of image[Xdn, i] whereas Xup refers to upper edge
    #Assuming no stray pixels other than our signal. That is all black pixels are joined together.
    Xup = np.argmin(gray[:, i]) #First occurence of black pixel
    Xdn = -np.argmin(np.flip(gray[:, i], axis = 0)) #Last occurence of black pixel
    if (math.fabs(Xup + Xdn) and Xdn is not None and Xup is not None): #Adding coz Xdn is literally negative
        gray[Xup: Xdn, i] = 0 #Joins signals if there is atleast one black pixel in the column

kernel = np.ones((2, 2), np.uint8)
gray = cv2.erode(cv2.bitwise_not(gray), kernel, iterations = 1)
gray = cv2.bitwise_not(gray)
cv2.imwrite('feeder.png', gray)
for i in range(gray.shape[1]):
    Xup = None
    Xdn = None
    Xup = np.argmin(gray[:, i])
    Xdn = -np.argmin(np.flip(gray[:, i], axis = 0))

    if (math.fabs(Xup - (Xdn + gray.shape[0]))<maxNoPeakLength):
        gray[Xup: Xdn, i] = 255
        gray[(Xup + gray.shape[0] + Xdn)//2, i] = 0

end = time.time()
cv2.imwrite('NewSig.png', gray)
print('Time taken: ', end - start)
