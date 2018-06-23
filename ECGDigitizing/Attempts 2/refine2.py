import cv2
import numpy
import sys
BLOCK_SIZE = 50
THRESHOLD = 25
def preprocess(image):
    image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return 255 - image
def postprocess(image):
    image = cv2.medianBlur(image, 5)
    image = cv2.medianBlur(image, 5)
    kernel = numpy.ones((5,5), numpy.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image
def get_block_index(image_shape, yx, block_size):
    y = numpy.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = numpy.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return numpy.meshgrid(y, x)
def adaptive_median_threshold(img_in):
    med = numpy.median(img_in)
    img_out = numpy.zeros_like(img_in)
    img_out[img_in - med < THRESHOLD] = 255
    return img_out
def block_image_process(image, block_size):
    out_image = numpy.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx])    
    return out_image
def process_image_file(filename):
    '''
   large = cv2.imread('1.jpg')
   rgb = cv2.pyrDown(large)
   small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
   grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

   _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
   connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
   # using RETR_EXTERNAL instead of RETR_CCOMP
   contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

   mask = np.zeros(bw.shape, dtype=np.uint8)

   for idx in range(len(contours)):
       x, y, w, h = cv2.boundingRect(contours[idx])
       mask[y:y+h, x:x+w] = 0
       cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
       r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if r > 0.45 and w > 8 and h > 8:
       cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

   cv2.imshow('rects', rgb) 
   '''
    image_in = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY) 
    cv2.imwrite('./TheBGR2GrayImage.png', image_in)  
    image_in = preprocess(image_in)
    image_out = block_image_process(image_in, BLOCK_SIZE)
    image_out = postprocess(image_out)    
    cv2.imwrite( './outOPEN.jpg', image_out)
   
if __name__ == "__main__":
    process_image_file('/Users/ishukalra/ECGDigitization/ECGDigitizing/Attempts(OhMyGoddess)/ECGNormal.png')