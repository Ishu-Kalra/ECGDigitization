import cv2
import numpy as np
from PIL import Image
from PIL import EpsImagePlugin
import pytesseract
import math
import re
import time
import timeit #@@TODO Check efficiency through this library

start = time.time()

digitize = cv2.imread('ECGSignalSnip.png') #Black background and white signal. So black pixel is [0 0 0] and white is some value greater than 200 probably. It ranges from [13 12 13] to [255 255 255]. Probably keep some high value to eliminate the noise on the edges of the signal skeleton. Or it is probably better better to use white background and black pixels ...
digitize = cv2.bitwise_not(digitize)
gray = cv2.cvtColor(digitize, cv2.COLOR_BGR2GRAY) #Much faster. Takes 2.5s in single channel as compared to 27.9s in 3 channels image
cv2.imwrite('SingleChannelSignal.png', gray)
start = time.time()
thresholdValue = 10 #Will split the pixel to black or white based on this
maxNoPeakLength = 20 #Maximum diiference that will tell no peak otherwise a peak. If less than this then no peak
for i in range(gray.shape[1]): #Column by column scanning
    gray[:, i] = (gray[:, i]>thresholdValue) * 255 #Thresholding. All pixels > thresholdValue are converted to white and rest to black

cv2.imwrite('AfterSigThresholding.png', gray)

end = time.time()
print('Time elapsed', end - start)
print('Shape of sig', gray.shape)

cv2.imwrite('NewSig.png', gray)

peakFinder = []
peaks = []

gray = cv2.imread('AfterSigThresholding.png')
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) #necessary to always convert when you read from imread. IMREAD automatically makes it 3 channels
maxNoPeakLength = 50.0
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
    if (math.fabs(Xup - (Xdn + gray.shape[0])) < maxNoPeakLength and Xdn is not None and Xup is not None): #Adding coz Xdn is literally negative
        gray[Xup: Xdn, i] = 0 #Joins signals if there is atleast one black pixel in the column
    elif (np.count_nonzero(gray[:, i]==0) > 0):
        peakFinder.append(i)

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
        gray[Xup: Xdn, i] = 255  #Whiten all except
        gray[(Xup + gray.shape[0] + Xdn)//2, i] = 0  #the average

counter = 0
print('PeakFinder', peakFinder)
print('\n')
for peakColumn in peakFinder:
    counter += 1
    #gray[:, peakColumn] = 127
    Xup = None
    Xdn = None
    Xup = np.argmin(gray[:, peakColumn]) #First occurence of black pixel
    Xdn = -np.argmin(np.flip(gray[:, peakColumn], axis = 0)) #Last occurence of black pixel
    XupNext = None
    XdnNext = None
    XupNext = np.argmin(gray[:, peakColumn + 1])
    XdnNext = -np.argmin(np.flip(gray[:, peakColumn + 1], axis = 0))
    #print('For peak column', peakColumn); print('XupNext and XdnNext are', XupNext, XdnNext); print('Sum NEXT IS', math.fabs(XupNext + XdnNext)); #print('Xup and Xdn are', Xup, Xdn); print('Sum SAME is', math.fabs(Xup + Xdn))
    #print(math.fabs(XupNext + XdnNext) < maxNoPeakLength)
    if (math.fabs(XupNext - (XdnNext + gray.shape[0])) < maxNoPeakLength): #peak completed
        peaks.append((peakColumn - counter + 1, peakColumn)) #Tuple as a peak coz it should be immutable
        counter = 0


print(peaks)
cleanedPeaks = []
#BUG BUSTING
#@@TODO Remove those ridiculous math.fabs statements, those print statements, those those Xup and Xdn. Find out why 91 or other such columns are getting appended.

for peak in peaks:
    if peak[0] != peak[1]:
        cleanedPeaks.append(peak)
        #gray[:, peak[0]] = 127
        #gray[:, peak[1]] = 127

print(" \n \n \n \n "); print(cleanedPeaks)

for val in cleanedPeaks:
    #im = gray[:, val[0]: val[1]] #Part of the image with only peaks
    for j in range(gray.shape[0]): #Row wise processing
        
        Xup = np.argmin(gray[j, val[0]: val[1]  + 1]) #All columns but one row. Exactly opposite as before
        Xdn = -np.argmin(np.flip(gray[j, val[0] : val[1] + 1], axis = 0))
        if (np.count_nonzero(gray[j, Xup + val[0]: Xdn + val[1] + 1] == 0)): #Checking only between the regions
            gray[j, Xup + val[0]: Xdn + val[1] + 1] = 255 #White them except the edges as done below
            gray[j, Xup + val[0]] = 0
            gray[j, Xdn  + val[1]] = 0


        
end = time.time()
cv2.imwrite('NewSig.png', gray)
print('Time taken: ', end - start)
