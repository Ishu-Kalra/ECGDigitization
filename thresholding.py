import cv2
import numpy as np
from PIL import Image
from PIL import EpsImagePlugin
import pytesseract
import math
import re
import time

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