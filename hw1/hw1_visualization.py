## Homework 1
##
## For this assignment, you will implement basic convolution, denoising,
## downsampling, and edge detection operations.
## Your implementation should be restricted to using low-level primitives
## in numpy (e.g. you may not call a Python library routine for convolution in
## the implementation of your convolution function).
##
## This notebook provides examples for testing your code.
## See hw1.py for detailed descriptions of the functions you must implement.

import numpy as np
import matplotlib.pyplot as plt

from util import *
from hw1 import *
from hw1_reference import * # delete me TODO 
ref = __import__('hw1_reference')

'''

# Problem 1 - Convolution (10 Points)
#
# Implement the conv_2d() function as described in hw1.py.
#
# The example below tests your implementation by convolving with a box filter.

image = load_image('data/69015.jpg')
box = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])
img = conv_2d(image,box) # TODO: these tests could be broken because both refr same file 
img1 = conv_2d(image,box)


print("int")
print(image)
print("corr")
print(img1)
print("mine")
print(img)
#print(np.max(img - img1))
#print(np.mean(img - img1))


# TODO: uncomment me out in the future
plt.figure(); plt.imshow(image, cmap='gray')
plt.figure(); plt.imshow(img, cmap='gray')
plt.figure(); plt.imshow(img1, cmap='gray')
plt.show()
'''
## Problem 2
# (a) Denoising with Gaussian filtering (5 Points)
##
## Implement denoise_gaussian() as described in hw1.py.
##
## The example below tests your implementation.
'''
image = load_image('data/148089_noisy.png')
imgA  = denoise_gaussian(image, 1.0) #TODO toying around here TODO : FIX 2.5 case instance
imgB  = denoise_gaussian(image, 1.0) # examples were 1.0 and 2.5 here

# TODO delete me
#print(np.max(imgA - imgB),np.min(imgA - imgB))
#print(np.mean(imgA - imgB))

plt.figure(); plt.imshow(image, cmap='gray')
plt.figure(); plt.imshow(imgA, cmap='gray')
plt.figure(); plt.imshow(imgB, cmap='gray')
plt.show()

'''



# (b) Denoising with bilateral filtering (5 Points)
##
## Implement denoise_bilateral() as described in hw1.py.
##
## The example below tests your implementation.

'''
image = load_image('data/148089_noisy.png')
imgA  = denoise_bilateral(image, sigma_s = 1.0, sigma_r=25.5)
imgB  = denoise_bilateral(image, sigma_s = 3.0, sigma_r=25.5)
#imgC  = denoise_bilateral(image, sigma_s = 5.0, sigma_r=25.5)
#imgc  = denoise_bilateral(image, sigma_s = 3.0, sigma_r=27.5)

#plt.figure(); plt.imshow(image, cmap='gray')
plt.figure(); plt.imshow(imgA, cmap='gray')
plt.figure(); plt.imshow(imgB, cmap='gray')
#plt.figure(); plt.imshow(imgC, cmap='gray')
plt.show()

raise
'''
## Problem 3 - Smoothing and downsampling (5 Points)
##
## Implement smooth_and_downsample() as described in hw1.py.
##
## The example below tests your implementation.
'''
image  = load_image('data/143090.jpg')
ds_image = smooth_and_downsample(image, downsample_factor = 2)
ds1_image = smooth_and_downsample(image, downsample_factor = 3)
ds2_image = smooth_and_downsample(image, downsample_factor = 4)


# downsample factor

plt.figure(); plt.imshow(image)
plt.figure(); plt.imshow(ds_image)
plt.figure(); plt.imshow(ds1_image)
plt.figure(); plt.imshow(ds2_image)
plt.show()

raise'''


## Problem 4 - Bilinear upsampling (5 Points)
##
## Implement bilinear_upsampling() as described in hw1.py.
##
## The example below tests your implementation.

'''
image  = load_image('data/69015.jpg')
us_image = bilinear_upsampling(image, upsample_factor = 2)

us1_image = bilinear_upsampling(image, upsample_factor = 3)

plt.figure(); plt.imshow(image)
plt.figure(); plt.imshow(us_image)
plt.figure(); plt.imshow(us1_image)
plt.show()
raise'''

'''
## Problem 5 - Sobel gradient operator (5 Points)
##
## Implement sobel_gradients() as described in hw1.py.
##
## The example below tests your implementation.

image  = load_image('data/69015.jpg')
dx, dy = ref.sobel_gradients(image)
dx1, dy1 = sobelaaa_gradients(image)
print("image")
print(image)
print("correct dx")
print(dx)
print("my dx")
print(dx1)
print("overall dif")
print(dx - dx1)

plt.figure(); plt.imshow(image, cmap='gray')
plt.figure(); plt.imshow(dx, cmap='gray')
plt.figure(); plt.imshow(dx1, cmap='gray')
plt.show()
'''
# Problem 6 -  (a) Nonmax suppression (10 Points)
#              (b) Edge linking and hysteresis thresholding (10 Points)
#              (c) Canny edge detection (5 Points)
#
# Implement nonmax_suppress(), hysteresis_edge_linking(), canny() as described in hw1.py
#
# The examples below test your implementation

#image  = load_image('data/edge_img/ben-simmons-wins-rookie-of-the-yearjpg.jpg')
#image  = load_image('data/edge_img/checker.png')
image  = load_image('data/edge_img/medium/001.jpg')
mag, nonmax, edge = canny(image,downsample_factor=[1,2,3,4,5,6,7,8,9,10]) #check for 2
plt.figure(); plt.imshow(image, cmap='gray')
plt.figure(); plt.imshow(mag, cmap='gray')
plt.figure(); plt.imshow(nonmax, cmap='gray')
plt.figure(); plt.imshow(edge, cmap='gray')
plt.show()

# Extra Credits:
# (a) Improve Edge detection image quality (5 Points)

