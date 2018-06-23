#Said paper -> https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4652928/#ref10

import cv2
import numpy as np
from PIL import Image
from PIL import EpsImagePlugin
import pytesseract
import math
import re
#TODO: tempcoderunnerfile.py was being created whwn other ECGs data was being used and I switch back to ECGNormal.png. It was being printed with ECGNormal and hence was giving an error in the in built visual studio code compiler.

#24 bit to 8 bit grayscale conversion
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
im = cv2.imread('ECGNormal.png')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

gray = cv2.bitwise_not(gray)
cv2.imwrite('ECGGray.png', gray)
#Changing dpi to 300 to reduce computational cost
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#TODO Remove the reduntant files using os to save space. Also Tesseract recognized 00-000-00 in the 300 dpi image. Hence it worked better in low dpi image.
im = Image.open("ECGGray.png")
im.save("test-300.png", dpi=(300,300))

image = cv2.imread('test-300.png')
#Threshold function. From pyimagesearch -> https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
#Skew correction included
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.bitwise_not(gray)   Can change black and white to white and black. Basically flips pixels. Quick and Dirty. Coz numpy lol :D

blur = cv2.GaussianBlur(gray,(5,5),0) 
#@TODO: Try MedianBlur instead of Gaussian Blur
#Gaussian Blur removes noise @@TODO Removing major grid lines and converting them to points. To be decided whether to keep or not! 
retVal, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print('RETVAL', retVal)
#Binarizes the image. Removes small grid lines and preserves big ones.
cv2.imwrite('thresh.png', thresh)

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
 
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = -(90 + angle)
 
# otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = -angle

# rotate the image to deskew it
(h, w) = thresh.shape[:2]   #@@Was taking original image initially to correct it
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)  #@@Was taking original image to correct it

#Can print the angle rotated in the image itself. Skipping for now @@TODO: Decide whether to keep it or not
#cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
# show the output image
print("[INFO] skew angle: {:.3f}".format(angle))
cv2.imwrite('SkewCorrected.png', rotated)

#Contour Detection LEARNING
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imwrite('FindContours.png', im2)
cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)
cv2.imwrite('DrawContours.png', im2)
'''

#Contour Detection IMPLEMENTATION
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
print(rotated.shape)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
print(kernel)
connected = cv2.morphologyEx(rotated, cv2.MORPH_OPEN, kernel)
cv2.imwrite('CLOSING.png', connected)


#For tesseract white background is better; so...
CLOSINGWHITE = cv2.bitwise_not(connected) #Flipping the pixels. Black to White and White to Black
cv2.imwrite('CLOSINGWHITE.png', CLOSINGWHITE)


'''
cv2.imwrite('Column', rotated.shape[0][200])

for i in range(rotated.shape[0]):
    col, contours, hierarchy = cv2.findContours()
'''
textRecognizeImage = CLOSINGWHITE #Image to be passed for recognition. Currently passing CLOSINGWHITE.png

PossibleAnnotations = []
PossibleAnnotationsPositions= []
#Not using dict but two lists to preserve degeneracy

columnWidth = 60 #    Tried 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500.    50 gave best result
rowWidth = 60 #keeping it same as columnWidth for now
for i in range(0, (textRecognizeImage.shape[1] // columnWidth) * columnWidth, columnWidth):#Doing // to get Exact image columns. Will leave out few columns at the end but they are reduntant anyway.
    for j in range(0, (textRecognizeImage.shape[0] // rowWidth) * rowWidth, rowWidth):
        thick = textRecognizeImage[j: j+rowWidth, i: i+columnWidth] #A block of image.
        #im, contours, hierarchy = cv2.findContours(thick, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        text = pytesseract.image_to_string(thick)
        if text != '':
            PossibleAnnotations.append(text)
            PossibleAnnotationsPositions.append((j + rowWidth // 2, i + columnWidth // 2))
        print(text)

print('Final Texts detected ', PossibleAnnotations)
print('Final Positions of Textx so detected ', PossibleAnnotationsPositions)
print('Pixel width and height were', columnWidth)

#Dividing into blocks
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@TODO convert into functions and call them instaed of the sequential code here :/
#Using search and not match coz pytesseract :(
'''
matchObjectColumnOne = re.match(r'/[IiL1fl\'\"\(\)]{1,4}/i', text)
    matchObjectColumnTwo = re.match(r'/a?V\s*[123\'\"]*/i', text)
    matchObjectColumnThree = re.match(r'/\s*V\d*\'?\"?/i', text)
    matchObjectColumnFour = re.match(r'/\s*V\d*\'?\"?/i', text)
'''
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

if (len(BLOCK_X) != len(BLOCK_Y)):
    print('Somethings broken as block lists of x and y lengths don\'t match :o')
    break
else:
    for i in range(1, len(BLOCK_X)):
        block = textRecognizeImage[BLOCK_X[i-1]: BLOCK_X[i], BLOCK_Y[i-1]: BLOCK_Y[i]]
        ##SIGNAL DIGITIZATION: ladies and gentlemen, the moment we were waiting for so earnestly
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #@@TODO: Adapt this into a function
        #Scanning each pixel by pixel so it will quite slow. Cannot help it here.
        



















'''
Final Texts detected  ['0‘10', '-00(', 'pu-', 'VI', 'aVl', '5(\n\nﬂ', 'z:', 'J', 'n V', '?_‘z', 'hes', '12', 'l3', '=.—', '11V', "1' 4", 'Nil', 'll', 'l5-', '\\II.', 'n-HJ']
Final Positions of Textx so detected  [(1645, 245), (1645, 385), (1645, 595), (525, 875), (875, 875), (1645, 875), (1015, 1155), (1645, 1155), (1645, 1295), (1365, 1435), (1645, 1435), (525, 1645), (875, 1645), (1295, 1645), (1645, 1715), (105, 2415), (525, 2415), (595, 2415), (1645, 2555), (1085, 2695), (1645, 3045)]
Pixel width and height were 70

Final Texts detected  ["'l", 'H', 'r', 'n', 'nn', "IU'J", 'n', "'UU", '‘I C', '2', '‘1)“', 'JM', 'll', "'1)", 'IIIV', '4N', 'IE', 'V3', '=', '_,...-', 'V"', '‘1‘', 'V1', 'II', '.l‘', '‘. .é-', '_..‘', 'Iﬂl', '"l.']
Final Positions of Textx so detected  [(1275, 125), (1375, 125), (1625, 125), (1625, 225), (1625, 275), (1675, 275), (1625, 375), (1675, 425), (1675, 575), (575, 625), (275, 875), (1075, 925), (1625, 1125), (275, 1175), (1675, 1275), (1075, 1425), (1375, 1425), (875, 1625), (1325, 1675), (625, 2275), (125, 2375), (475, 2375), (875, 2375), (1625, 2375), (1425, 2425), (675, 2975), (575, 3075), (1625, 3075), (1675, 3075)]
Pixel width and height were 50

Final Texts detected  ["'l", 'w.“', "|I’)('", "000']", 'b', "H'J-GO", "'UU", "(-d:'\n\nIs.", 'mm', "II'5(\n\nﬂ", 'umb:', 'l’l\n\\', 'm V', 'm—', 'best', ':10', 'J/', 'V"', '‘1‘', 'V1', '\\JL.', 'U’Vrl', '_..‘', 'In-H):']
Final Positions of Textx so detected  [(1250, 150), (1450, 150), (1650, 150), (1650, 250), (250, 350), (1650, 350), (1650, 450), (1650, 650), (1650, 750), (1650, 850), (1650, 1050), (1650, 1150), (1650, 1250), (1350, 1450), (1650, 1450), (1650, 1550), (250, 1950), (150, 2350), (450, 2350), (850, 2350), (1450, 2450), (1650, 2450), (550, 3050), (1650, 3050)]
Pixel width and height were 100

Final Texts detected  ['10C', 'HUG', 'GOl', '=-', 'c d', 'u', 'I‘d', 'ru-', '5(\n\nﬂ', 'mb', 'n V', "f'he", '’Sl', ':H', 'I)', '11V', '| k:', 'ﬂ', 'J', "1'.", 'V1', "'JI", '51', ':I', 'l5-', 'In-l', '‘05']
Final Positions of Textx so detected  [(1650, 150), (1650, 270), (1650, 390), (570, 630), (1650, 630), (90, 870), (150, 870), (270, 870), (1650, 870), (1650, 1050), (1650, 1290), (1650, 1410), (1650, 1470), (1650, 1530), (90, 1590), (1650, 1710), (1470, 1950), (630, 2010), (390, 2190), (1410, 2190), (870, 2370), (1650, 2370), (510, 2430), (1350, 2430), (1650, 2550), (1650, 3030), (1650, 3090)]
Pixel width and height were 60
'''