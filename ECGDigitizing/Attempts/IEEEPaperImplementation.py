#Said paper -> https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4652928/#ref10

import cv2
import numpy
#24 bit to 8 bit grayscale conversion
im = cv2.imread('ECGNormal.png')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

cv2.imwrite('ECGGray.png', gray)