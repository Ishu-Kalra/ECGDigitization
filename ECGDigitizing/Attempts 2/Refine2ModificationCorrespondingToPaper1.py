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
   image_in = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)    
   image_in = preprocess(image_in)
   image_out = block_image_process(image_in, BLOCK_SIZE)
   image_out = postprocess(image_out)    
   cv2.imwrite( './out.jpg', image_out)
if __name__ == "__main__":
  process_image_file('./im.jpg')