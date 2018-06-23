import cv2
import numpy as np
from PIL import Image
from PIL import EpsImagePlugin
import pytesseract
import math
import re
import time
'''
im = cv2.imread('CLOSING.png')
textRecognizeImage = im
#Using 100X100 pixel window
PossibleAnnotations = ["'l", 'w.“', "|I’)('", "000']", 'b', "H'J-GO", "'UU", "(-d:'\n\nIs.", 'mm', "II'5(\n\nﬂ", 'umb:', 'l’l\n\\', 'm V', 'm—', 'best', ':10', 'J/', 'V"', '‘1‘', 'V1', '\\JL.', 'U’Vrl', '_..‘', 'In-H):']
PossibleAnnotationsPositions = [(1250, 150), (1450, 150), (1650, 150), (1650, 250), (250, 350), (1650, 350), (1650, 450), (1650, 650), (1650, 750), (1650, 850), (1650, 1050), (1650, 1150), (1650, 1250), (1350, 1450), (1650, 1450), (1650, 1550), (250, 1950), (150, 2350), (450, 2350), (850, 2350), (1450, 2450), (1650, 2450), (550, 3050), (1650, 3050)]

BLOCK_X = [0] #Begins at 0. Later end/break points will be added
BLOCK_Y = [0] #Begins at 0. Later end/break points will be added.
#Region between consecutive break points will constitute a block
#Column 1
for i in range(len(PossibleAnnotations)):
    #REGEX woohoo!
    text = PossibleAnnotations[i]
    searchObjectColumnOne = re.search(r'/[IiL1fl\'\"\(\)]{1,4}/i', text)
    if searchObjectColumnOne:
        BLOCK_X.append(PossibleAnnotationsPositions[i][0])
        BLOCK_Y.append(PossibleAnnotationsPositions[i][1])

#Column 2
for i in range(len(PossibleAnnotations)):
    #REGEX woohoo!
    text = PossibleAnnotations[i]
    searchObjectColumnTwo = re.search(r'/a?V\s*[123\'\"]*/i', text)
    if searchObjectColumnTwo:
        BLOCK_X.append(PossibleAnnotationsPositions[i][0])
        BLOCK_Y.append(PossibleAnnotationsPositions[i][1])

#Column 3 and 4
for i in range(len(PossibleAnnotations)):
    #REGEX woohoo!
    text = PossibleAnnotations[i]
    searchObjectColumnThirdAndFourth = re.search(r'/\s*V\d*\'?\"?/i', text)
    if searchObjectColumnThirdAndFourth:
        BLOCK_X.append(PossibleAnnotationsPositions[i][0])
        BLOCK_Y.append(PossibleAnnotationsPositions[i][1])

print('Final Block Lists X and Y', BLOCK_X, BLOCK_Y)
'''
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
for i in range(gray.shape[1]):
    Xup = None
    Xdn = None
    #Assuming no stray pixels other than our signal. That is all black pixels are joined together.
    Xup = np.argmin(gray[:, i]) #First occurence of black pixel
    Xdn = np.argmin(np.flip(gray[:, i], axis = 0)) #Last occurence of black pixel
    #if (math.fabs(Xup - Xdn) > maxNoPeakLength):
    #gray[Xdn: Xup, i] = 0 #Join them and make one whole black column

end = time.time()
print('Time elapsed', end - start)
print('Shape of sig', gray.shape)

cv2.imwrite('NewSig.png', gray)