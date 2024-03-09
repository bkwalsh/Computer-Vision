import numpy as np
import math
"""
   Mirror an image about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of the specified width created by mirroring the interior
"""
def mirror_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   # mirror top/bottom
   top    = image[:wx:,:]
   bottom = image[(sx-wx):,:]
   img = np.concatenate( \
      (top[::-1,:], image, bottom[::-1,:]), \
      axis=0 \
   )
   # mirror left/right
   left  = img[:,:wy]
   right = img[:,(sy-wy):]
   img = np.concatenate( \
      (left[:,::-1], img, right[:,::-1]), \
      axis=1 \
   )
   return img

# my helper function to mirror bottom, right dimensions for padding 
# to get exact image for upsample/downsample
def mirror_border_bottom_right(image, wx=1, wy=1):
    assert image.ndim == 2, 'image should be grayscale'
    sx, sy = image.shape
    
    # mirror bottom
    bottom = image[(sx-wx):, :]
    img = np.concatenate(
        (image, bottom[::-1, :]),
        axis=0
    )
    
    # mirror right
    right = img[:, (sy-wy):]
    img = np.concatenate(
        (img, right[:,::-1]),
        axis=1
    )
    
    return img


# helper function for fitting shapes for edge detecto
def trim_border_bottom_right(image, wx=1, wy=1):
    assert image.ndim == 2, 'image should be grayscale'
    sx, sy = image.shape
    
    # trim bottom
    img = image[:(sx-wx), :]
    
    # trim right
    img = img[:, :(sy-wy)]
    
    return img


"""
   Pad an image with zeros about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of zeros
"""
def pad_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.zeros((sx+2*wx, sy+2*wy))
   img[wx:(sx+wx),wy:(sy+wy)] = image
   return img

"""
   Remove the border of an image.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx - 2*wx, sy - 2*wy), extracted by
              removing a border of the specified width from the sides of the
              input image
"""
def trim_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.copy(image[wx:(sx-wx),wy:(sy-wy)])
   return img

"""
   Return an approximation of a 1-dimensional Gaussian filter.

   The returned filter approximates:

   g(x) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2) / (2 * sigma^2) )

   for x in the range [-3*sigma, 3*sigma]
"""
def gaussian_1d(sigma = 1.0):
   width = np.ceil(3.0 * sigma)
   x = np.arange(-width, width + 1)
   g = np.exp(-(x * x) / (2 * sigma * sigma))
   g = g / np.sum(g)          # normalize filter to sum to 1 ( equivalent
   g = np.atleast_2d(g)       # to multiplication by 1 / sqrt(2*pi*sigma^2) )
   return g

"""

My helper function for simple 1-d convolution helper 
function for sobel and denoise gauss 

"""
def oneDconvolution(image, kernel):
   iH, iW = image.shape
   kH, kW = kernel.shape
   rH = iH - kH + 1
   rW = iW - kW + 1
   result = np.zeros((rH, rW))

   for i in range(rH):
      for j in range(rW):
            points = image[i:i + kH, j:j + kW]
            result[i, j] = np.sum(np.multiply(points, kernel))

   return result


"""
   CONVOLUTION IMPLEMENTATION (10 Points)

   Convolve a 2D image with a 2D filter.

   Requirements:

   (1) Return a result the same size as the input image.

   (2) You may assume the filter has odd dimensions.

   (3) The result at location (x,y) in the output should correspond to
       aligning the center of the filter over location (x,y) in the input
       image.

   (4) When computing a product at locations where the filter extends beyond
       the defined image, treat missing terms as zero.  (Equivalently stated,
       treat the image as being padded with zeros around its border).

   You must write the code for the nested loops of the convolutions yourself,
   using only basic loop constructs, array indexing, multiplication, and
   addition operators.  You may not call any Python library routines that
   implement convolution.

   Arguments:
      image  - a 2D numpy array
      filt   - a 1D or 2D numpy array, with odd dimensions
      mode   - 'zero': preprocess using pad_border or 'mirror': preprocess using mirror_border.

   Returns:
      result - a 2D numpy array of the same shape as image, containing the
               result of convolving the image with filt
"""
def conv_2d(image, filt, mode='zero'):
   rows, cols = image.shape
   result = np.zeros((rows, cols))
   filt = np.atleast_2d(filt)
   filt = np.flip(filt)
   fX, fY = filt.shape
   rX, rY = fX // 2, fY // 2 

   if mode == 'zero':
      padded_image = pad_border(image,wx=rX,wy=rY)
   elif mode == 'mirror':
      padded_image = mirror_border(image,wx=rX,wy=rY)
   else: 
      raise Exception("Bad Mode input")
   
   for i in range(rows):
      for j in range(cols):
         region = padded_image[i:i + fX, j:j + fY]
         result[i, j] = np.sum(filt * region)

   return result

"""

   GAUSSIAN DENOISING (5 Points)

   Denoise an image by convolving it with a 2D Gaussian filter.

   Convolve the input image with a 2D filter G(x,y) defined by:

   G(x,y) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2 + y^2) / (2 * sigma^2) )

   You may approximate the G(x,y) filter by computing it on a
   discrete grid for both x and y in the range [-3*sigma, 3*sigma].

   See the gaussian_1d function for reference.

   Note:
   (1) Remember that the Gaussian is a separable filter.
   (2) Denoising should not create artifacts along the border of the image.
       Make an appropriate assumption in order to obtain visually plausible
       results along the border.

   Arguments:
      image - a 2D numpy array
      sigma - standard deviation of the Gaussian

   Returns:
      img   - denoised image, a 2D numpy array of the same shape as the input
"""

def denoise_gaussian(image, sigma = 1.0):
   g = gaussian_1d(sigma)
   return conv_2d(conv_2d(image, g, mode="mirror"), g.T, mode="mirror") 


"""
    BILATERAL DENOISING (5 Points)
    Denoise an image by applying a bilateral filter
    Note:
        Performs standard bilateral filtering of an input image.
        Reference link: https://en.wikipedia.org/wiki/Bilateral_filter

        Basically, the idea is adding an additional edge term to Guassian filter
        described above.

        The weighted average pixels:

        BF[I]_p = 1/(W_p)sum_{q in S}G_s(||p-q||)G_r(|I_p-I_q|)I_q

        In which, 1/(W_p) is normalize factor, G_s(||p-q||) is spatial Guassian
        term, G_r(|I_p-I_q|) is range Guassian term.

        We only require you to implement the grayscale version, which means I_p
        and I_q is image intensity.

    Arguments:
        image       - input image
        sigma_s     - spatial param (pixels), spatial extent of the kernel,
                       size of the considered neighborhood.
        sigma_r     - range param (not normalized, a propotion of 0-255),
                       denotes minimum amplitude of an edge
    Returns:
        img   - denoised image, a 2D numpy array of the same shape as the input
"""

def denoise_bilateral(image, sigma_s=1, sigma_r=25.5):
   assert image.ndim == 2, 'image should be grayscale'

   # local fn for calculating w_{ijkl} term from wikipedia page
   def w_term(i, j, k, l, ijImage, klImage):
      first_term = - (((i - k) ** 2) + ((j - l) ** 2)) / (2 * (sigma_s ** 2))
      second_term = ((ijImage - klImage) ** 2) / (2 * (sigma_r ** 2))
      return math.exp(first_term - second_term)
   
   rS = math.ceil(3 * sigma_s)
   rows, cols = image.shape
   img = np.zeros_like(image)
   padded = mirror_border(image, wx = rS, wy = rS)

   for i in range(rS, rows + rS):
      for j in range(rS, cols + rS):
         pointRange = padded[i - rS: i + rS + 1, j - rS: j + rS + 1]
         ijImage = pointRange[rS, rS]
         top, bottom = 0,0 

         for k in range((rS * 2) + 1):
            for l in range((rS * 2) + 1):
               kl_term = pointRange[k,l]
               w = w_term(i, j, (i + k - rS), (j + l - rS), ijImage, kl_term) 
               bottom += w
               top += w * kl_term

         img[i - rS, j - rS] = top / bottom

   return img

"""
   SMOOTHING AND DOWNSAMPLING (5 Points)

   Smooth an image by applying a gaussian filter, followed by downsampling with a factor k.

   Note:
      Image downsampling is generally implemented as two-step process:

        (1) Smooth images with low pass filter, e.g, gaussian filter, to remove
            the high frequency signal that would otherwise causes aliasing in
            the downsampled outcome.

        (2) Downsample smoothed images by keeping every kth samples.

      Make an appropriate choice of sigma to avoid insufficient or over smoothing.

      In principle, the sigma in gaussian filter should respect the cut-off frequency
      1 / (2 * k) with k being the downsample factor and the cut-off frequency of
      gaussian filter is 1 / (2 * pi * sigma).


   Arguments:
     image - a 2D numpy array
     downsample_factor - an integer specifying downsample rate

   Returns:
     result - downsampled image, a 2D numpy array with spatial dimension reduced
"""
def smooth_and_downsample(image, downsample_factor = 2):
   rows, cols = image.shape
   dsR, dsC = rows // downsample_factor, cols // downsample_factor
   result = np.zeros((dsR, dsC)) 
   gauss_Sigma = 1 / (2 * downsample_factor * np.pi)
   smoothed = denoise_gaussian(image, sigma = gauss_Sigma) 

   for i in range(0, rows):
      for j in range(0, cols):
         ii,jj = i // downsample_factor, j // downsample_factor
         if ii >= dsR or jj >= dsC:
            continue
         if (i % downsample_factor == 0) and (j % downsample_factor == 0):
            result[ii,jj] = smoothed[i,j] 

   return result


"""
   BILINEAR UPSAMPLING (5 Points)

   Upsampling the input images with a factor of k with bilinear kernel

   Note:
      Image upsampling is generally implemented by mapping each output pixel
      (x_out,y_out) onto input images coordinates (x, y) = (x_out / k, y_out / k).
      Then, we use bilinear kernel to compute interpolated color at pixel
      (x,y), which requires to round (x, y) to find 4 neighboured pixels:

          P11 = (x1, y1)      P12 = (x1, y2)
          P21 = (x2, y1)      P22 = (x2, y2)

      where
          x1 = floor(x / k),  y1 = floor(y / k),
          x2 = ceil (x / k),  y2 = ceil (y / k)

      In practice, you can simplify the 2d interpolation formula by applying 1d
      interpolation along each axis:

          # interpolate along x axis
          C(x,y1) = (x2 - x)/(x2 - x1) * C(x1, y1) +  (x - x1)/(x2 - x1) * C(x2, y1)
          C(x,y2) = (x2 - x)/(x2 - x1) * C(x1, y2) +  (x - x1)/(x2 - x1) * C(x2, y2)

          # interpolate along y axis
          C(x,y) =  (y2 - y)/(y2 - y1) * C(x, y1)  +  (y - y1)/(y2 - y1) * C(x, y2)

      where C(x,y) denotes the pixel color at (x,y).

   Arguments:
     image - a 2D numpy array
     upsample_factor - an integer specifying upsample rate

   Returns:
     result - upsampled image, a 2D numpy array with spatial dimension increased
"""
def bilinear_upsampling(image, upsample_factor):
    rows, columns = image.shape
    image = mirror_border(image)
    nRows = rows * upsample_factor
    nCols = columns * upsample_factor
    result = np.zeros((nRows, nCols), dtype = image.dtype)

    for y_out in range(nRows):
        for x_out in range(nCols):
            x = x_out / upsample_factor
            y = y_out / upsample_factor

            x1, y1 = int(np.floor(x)), int(np.floor(y))
            x2, y2 = int(np.ceil(x)), int(np.ceil(y))

            dx1, dx2 = x - x1, x2 - x
            dy1, dy2 = y - y1, y2 - y

            w1 = dx2 * dy2
            w2 = dx1 * dy2
            w3 = dx2 * dy1
            w4 = dx1 * dy1

            total_weight = w1 + w2 + w3 + w4
            
            if total_weight != 0:
               w1 /= total_weight
               w2 /= total_weight
               w3 /= total_weight
               w4 /= total_weight

            # NOTE: manually handling the corner case of total_weight = 0 
            # with default even distribution across 4 points
            else:
               w1,w2,w3,w4 = .25, .25, .25, .25

            result[y_out, x_out] = w1 * image[y1, x1] + w2 * image[y1, x2] \
                 + w3 * image[y2, x1] + w4 * image[y2, x2]
            
   
    return result 


"""
   SOBEL GRADIENT OPERATOR (5 Points)
   Compute an estimate of the horizontal and vertical gradients of an image
   by applying the Sobel operator.
   The Sobel operator estimates gradients dx(horizontal), dy(vertical), of
   an image I as:

         [ 1  0  -1 ]
   dx =  [ 2  0  -2 ] (*) I
         [ 1  0  -1 ]

         [  1  2  1 ]
   dy =  [  0  0  0 ] (*) I
         [ -1 -2 -1 ]

   where (*) denotes convolution.
   Note:
      (1) Your implementation should be as efficient as possible.
      (2) Avoid creating artifacts along the border of the image.
   Arguments:
      image - a 2D numpy array
   Returns:
      dx    - gradient in x-direction at each point
              (a 2D numpy array, the same shape as the input image)
      dy    - gradient in y-direction at each point
              (a 2D numpy array, the same shape as the input image)
"""
def sobel_gradients(image):
   cB = np.array([[1, 0, -1]])
   cA = np.array([[1, 2, 1]])
   dx = conv_2d(conv_2d(image, cB, mode = "mirror"), cA.T, mode = "mirror")
   dy = conv_2d(conv_2d(image, cA, mode = "mirror"), cB.T, mode = "mirror")

   return dx,dy


"""
   NONMAXIMUM SUPPRESSION (10 Points)

   Nonmaximum suppression.

   Given an estimate of edge strength (mag) and direction (theta) at each
   pixel, suppress edge responses that are not a local maximum along the
   direction perpendicular to the edge.

   Equivalently stated, the input edge magnitude (mag) represents an edge map
   that is thick (strong response in the vicinity of an edge).  We want a
   thinned edge map as output, in which edges are only 1 pixel wide.  This is
   accomplished by suppressing (setting to 0) the strength of any pixel that
   is not a local maximum.

   Note that the local maximum check for location (x,y) should be performed
   not in a patch surrounding (x,y), but along a line through (x,y)
   perpendicular to the direction of the edge at (x,y).

   A simple, and sufficient strategy is to check if:
      ((mag[x,y] > mag[x + ox, y + oy]) and (mag[x,y] >= mag[x - ox, y - oy]))
   or
      ((mag[x,y] >= mag[x + ox, y + oy]) and (mag[x,y] > mag[x - ox, y - oy]))
   where:
      (ox, oy) is an offset vector to the neighboring pixel in the direction
      perpendicular to edge direction at location (x, y)

   Arguments:
      mag    - a 2D numpy array, containing edge strength (magnitude)
      theta  - a 2D numpy array, containing edge direction in [0, 2*pi)

   Returns:
      nonmax - a 2D numpy array, containing edge strength (magnitude), where
               pixels that are not a local maximum of strength along an
               edge have been suppressed (assigned a strength of zero)
"""
def nonmax_suppress(mag, theta):
   rows, cols = mag.shape
   nonmax = np.copy(mag)
   padMag = pad_border(mag)

   for i in range(0, rows):
      for j in range(0, cols):

         direction = theta[i,j]
         ox = int(np.round(np.cos(direction)))
         oy = int(np.round(np.sin(direction)))
         x,y = i + 1,j + 1

         if ((padMag[x,y] >= padMag[x + ox, y + oy]) 
                  and (padMag[x,y] >= padMag[x - ox, y - oy])):
            continue
         else:
            nonmax[i,j] = 0 

   return nonmax

"""
   HYSTERESIS EDGE LINKING (10 Points)

   Hysteresis edge linking.

   Given an edge magnitude map (mag) which is thinned by nonmaximum suppression,
   first compute the low threshold and high threshold so that any pixel below
   low threshold will be thrown away, and any pixel above high threshold is
   a strong edge and will be preserved in the final edge map.  The pixels that
   fall in-between are considered as weak edges.  We then add weak edges to
   true edges if they connect to a strong edge along the gradient direction.

   Since the thresholds are highly dependent on the statistics of the edge
   magnitude distribution, we recommend to consider features like maximum edge
   magnitude or the edge magnitude histogram in order to compute the high
   threshold.  Heuristically, once the high threshod is fixed, you may set the
   low threshold to be propotional to the high threshold.

   Note that the thresholds critically determine the quality of the final edges.
   You need to carefully tuned your threshold strategy to get decent
   performance on real images.

   For the edge linking, the weak edges caused by true edges will connect up
   with a neighbouring strong edge pixel.  To track theses edges, we
   investigate the 8 neighbours of strong edges.  Once we find the weak edges,
   located along strong edges' gradient direction, we will mark them as strong
   edges.  You can adopt the same gradient checking strategy used in nonmaximum
   suppression.  This process repeats util we check all strong edges.

   In practice, we use a queue to implement edge linking.  In python, we could
   use a list and its fuction .append or .pop to enqueue or dequeue.

   Arguments:
     nonmax - a 2D numpy array, containing edge strength (magnitude) which is thined by nonmaximum suppression
     theta  - a 2D numpy array, containing edeg direction in [0, 2*pi)

   Returns:
     edge   - a 2D numpy array, containing edges map where the edge pixel is 1 and 0 otherwise.
"""

def hysteresis_edge_linking(nonmax, theta):
   result = np.zeros_like(nonmax)
   rows, cols = nonmax.shape
   padMag = pad_border(nonmax)
   flat = nonmax.flatten()
   tX,tY = theta.shape

   # following otsu method for finding best threshhold for an image  
   # https://en.wikipedia.org/wiki/Otsu%27s_method
   H, _ = np.histogram(flat, bins = 256, range = [0, 256])   
   O = np.zeros(256)
   M = np.zeros(256)   
   P = H / np.sum(H)
   O[0] = P[0]
   M[0] = 0
   for i in range(1, 256):
      O[i] = O[i-1] + P[i]
      M[i] = M[i-1] + i * P[i]
   gM = M[255]
   sBS = (gM * O - M) ** 2 / (O * (1 - O))
   optimal = np.argmax(sBS)
   threshhold = optimal / 255.0 * 10.0

   # thresholding
   # NOTE: in threshold testing, considering real test images
   # because many of the edges are equal for synthetic image
   # checker.png, a higher threshhold leads to better performance
 
   #threshhold = 9 # <- uncomment me for checker.png testing
   lowerThreshhold = np.percentile(flat, threshhold)
   upperThreshhold = np.percentile(flat, 100-threshhold)
   queue = []

   for i in range(0,rows):
      for j in range(0,cols):

         if nonmax[i,j] <= lowerThreshhold:
            result[i,j] = 0
         elif nonmax[i,j] >= upperThreshhold:
            result[i,j] = 1
            queue.append((i,j))

   while queue:
      (i,j) = queue.pop(0)
      neighbors = range(-1, 2)
      for x in neighbors:
         for y in neighbors:

            if (x,y) == (0,0) \
               or (0 <= (i + x) <= tX) \
               or (0 <= (j + y) <= tY) \
               or (result[x,y] == 1):
               continue
            else:
               d = theta[i,j]
               ox = int(np.round(np.cos(d)))
               oy = int(np.round(np.sin(d)))
               xx,yy = x + 1,y + 1

               if ((padMag[xx,yy] >= padMag[xx + ox, yy + oy]) 
                     and (padMag[xx,yy] >= padMag[xx - ox, yy - oy])):
                  result[x,y] = 1 
                  queue.append((x,y))

   return result


"""
   CANNY EDGE DETECTOR (5 Points)

   Canny edge detector.

   Given an input image:

   (1) Compute gradients in x- and y-directions at every location using the
       Sobel operator.  See sobel_gradients() above.

   (2) Estimate edge strength (gradient magnitude) and direction.

   (3) Run (1)(2) on downsampled images with multiple factors and
       then combine the results via upsampling to original resolution.

   (4) Perform nonmaximum suppression of the edge strength map, thinning it
       in the direction perpendicular to that of a local edge.
       See nonmax_suppress() above.

   (5) Compute the high threshold and low threshold of edge strength map
       to classify the pixels as strong edges, weak edges and non edges.
       Then link weak edges to strong edges

   Return the original edge strength estimate (max), the edge
   strength map after nonmaximum suppression (nonmax) and the edge map
   after edge linking (edge)

   Arguments:
      image             - a 2D numpy array
      downsample_factor - a list of interger

   Returns:
      mag      - a 2D numpy array, same shape as input, edge strength at each pixel
      nonmax   - a 2D numpy array, same shape as input, edge strength after nonmaximum suppression
      edge     - a 2D numpy array, same shape as input, edges map where edge pixel is 1 and 0 otherwise.
"""
def canny(image, downsample_factor = [1]):
   # local func to calculate edge strength and direction given
   # gradX and gradY (calculated by sobel here)
   def strAndDir(gradients_x, gradients_y):
      edgeStr = np.sqrt(gradients_x**2 + gradients_y**2)
      edgeDir = np.arctan2(gradients_y, gradients_x)
      return edgeStr, edgeDir

   gradX, gradY = sobel_gradients(image)  
   mag, theta = strAndDir(gradX, gradY) 

   for factor in downsample_factor:
      # getting distance from factor if uneven downsample divison
      rX, rY = image.shape[0] % factor, image.shape[1] % factor
      if rX != 0:
         rX = factor - rX
      if rY != 0:
         rY = factor - rY

      # mirroring up to nearest divisable shape so image divides perfectly
      pImage = mirror_border_bottom_right(image,wx = rX,wy = rY)

      ds = smooth_and_downsample(pImage, downsample_factor = factor)
      gradXDS, gradYDS = sobel_gradients(ds)
      edgeStrDS, edgeDirDS = strAndDir(gradXDS, gradYDS)

      edgeStrUS = bilinear_upsampling(edgeStrDS, upsample_factor = factor)
      edgeDirUS = bilinear_upsampling(edgeDirDS, upsample_factor = factor)

      # cropping back down by reminder to get same shape as initial image
      # before addition step
      edgeStrUS = trim_border_bottom_right(edgeStrUS,wx=rX,wy=rY)
      edgeDirUS = trim_border_bottom_right(edgeDirUS,wx=rX,wy=rY)

      mag += edgeStrUS 
      theta += edgeDirUS

   nonmax = nonmax_suppress(mag, theta) 
   edge = hysteresis_edge_linking(nonmax, theta) 

   return mag,nonmax,edge