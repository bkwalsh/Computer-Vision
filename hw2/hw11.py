import numpy as np
import numpy.linalg as linalg
# Kameel Khabaz
# Homework 1 CMSC 25040 
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
   Return the value of a Gaussian for a specific input, centerd at 0 with max peak of 1.
"""
def gaussian(x, sigma):
   return np.exp(-(x * x) / (2 * sigma * sigma))

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
   # make sure that both image and filter are 2D arrays
   assert image.ndim == 2, 'image should be grayscale'
   filt = np.atleast_2d(filt)
   ##########################################################################
   # must flip filter by definition of convolution
   filt = np.flip(filt)
   img_r, img_c = image.shape
   filt_r, filt_c = filt.shape
   pad_x = filt_r // 2
   pad_y = filt_c // 2

   result = np.zeros((img_r,img_c))
   if mode == "zero":
      padded_image = pad_border(image, wx = pad_x, wy = pad_y)
   elif mode == "mirror":
      padded_image = mirror_border(image, wx = pad_x, wy = pad_y)
      
   for i in range(img_r):
      for j in range(img_c):
         img_cutout = padded_image[i: i + filt_r, 
                                   j: j + filt_c]
         result[i, j] = np.sum(img_cutout * filt)

   ##########################################################################
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
   ##########################################################################
   disc_gaussian_1d = gaussian_1d(sigma)
   img_conv1 = conv_2d(image, disc_gaussian_1d, mode="mirror")
   ##########################################################################
   return conv_2d(img_conv1, disc_gaussian_1d.T, mode="mirror") 

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
   ##########################################################################
   img_r, img_c = image.shape
   spatial_width = int(np.ceil(2 * sigma_s)) + 1
   pad = spatial_width // 2

   result = np.zeros((img_r,img_c))
   padded_image = mirror_border(image, wx = pad, wy = pad)
      
   for i in range(img_r):
      for j in range(img_c):
         img_cutout = padded_image[i: i + spatial_width, 
                                   j: j + spatial_width]

         # apply bilateral filter to pixel
         I_x = image[i, j] # main point intensity
         Ifiltered_x = 0 # resulting intensity post-filter
         weight = 0 # total weight
         for k in range(spatial_width):
            for l in range(spatial_width):
               I_xi = img_cutout[k, l]
               w_i = gaussian(I_xi - I_x, sigma_r) * gaussian(linalg.norm([k, l]), sigma_s)
               Ifiltered_x += I_xi * w_i
               weight += w_i

         result[i, j] = Ifiltered_x / weight

   ##########################################################################
   return result

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
def smooth_and_downsample1(image, downsample_factor = 2):
    ##########################################################################
    sigma = downsample_factor / np.pi
    smoothed_image = denoise_gaussian(image, sigma)
    ##########################################################################
    return smoothed_image[::downsample_factor, ::downsample_factor]

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
def bilinear_upsampling1(image, upsample_factor = 2):
    ##########################################################################
    if upsample_factor == 1:
      return image

    img_r, img_c = image.shape
    result = np.zeros([img_r * upsample_factor, img_c * upsample_factor])
    uimg_r, uimg_c = result.shape
    for x_out in range(uimg_r):
      for y_out in range(uimg_c):
         x = x_out / upsample_factor
         y = y_out / upsample_factor

         y1 = np.floor(y).astype('int')
         y2 = np.ceil(y).astype('int')
         # check for edge case 
         if y2 == img_c:
            y2 = y1

         x1 = np.floor(x).astype('int')
         x2 = np.ceil(x).astype('int')
         # check for edge case
         if x2 == img_r:
            x2 = x1 

         if x1 == x2 and y1 == y2:
            # corresponds to point in original image
            result[x_out, y_out] = image[int(x), int(y)]
         elif x1 == x2:
            # xs match, interp along y
            result[x_out, y_out] = (y2 - y)/(y2 - y1) * image[int(x), y1]  +  \
                                    (y - y1)/(y2 - y1) * image[int(x), y2]
         elif y1 == y2:
            # ys match, interp along x
            result[x_out, y_out] = (x2 - x)/(x2 - x1) * image[x1, int(y)] +  \
                                    (x - x1)/(x2 - x1) * image[x2, int(y)]
         else:
            C_x_y1 = (x2 - x)/(x2 - x1) * image[x1, y1] +  (x - x1)/(x2 - x1) * image[x2, y1]
            C_x_y2 = (x2 - x)/(x2 - x1) * image[x1, y2] +  (x - x1)/(x2 - x1) * image[x2, y2]
            result[x_out, y_out] =  (y2 - y)/(y2 - y1) * C_x_y1  +  (y - y1)/(y2 - y1) * C_x_y2

    ##########################################################################
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
def sobel_gradients1(image):
   ##########################################################################
   # Sobel filter is separable
   dx = conv_2d(image, [1, 0, -1], mode="mirror")
   dx = conv_2d(dx, np.reshape([1, 2, 1],(-1, 1)) , mode = "mirror")
   dy = conv_2d(image, [1, 2, 1], mode="mirror")
   dy = conv_2d(dy, np.reshape([1, 0, -1],(-1, 1)), mode = "mirror")
   ##########################################################################
   return dx, dy

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
   ##########################################################################
   img_r, img_c = mag.shape
   nonmax = np.zeros([img_r, img_c])
   for y in range(1, img_r - 1):
      for x in range(1, img_c - 1):
         # check for edge direction, perform nonmax suppression 
         # in orthogonal direction 
         if abs(np.sin(theta[y,x])) < np.sin(np.pi / 8):
            # horizontal edge, vertical suppression
            ox, oy = 0, 1
         elif abs(np.sin(theta[y,x])) > np.sin(3 * np.pi / 8):
            # vertical edge, horizontal suppression
            ox, oy = 1, 0
         elif np.cos(theta[y,x]) > 0:
            # diagonal edge in pi/4 direction, suppress other diagonal
            ox, oy = -1, 1
         else:
            # diagonal edge in 3 pi/4 direction, suppress other diagonal
            ox, oy = 1, 1

         p1y = y + oy
         p1x = x + ox

         p2y = y - oy
         p2x = x - ox

         # suppress non-max edge, check 2 formulations and confirm bounds are correct
         if   mag[y,x] >  mag[p1y, p1x] and mag[y,x] >= mag[p2y, p2x]:
            nonmax[y, x] = mag[y, x]
         elif mag[y,x] >= mag[p1y, p1x] and mag[y,x] >  mag[p2y, p2x]:
            nonmax[y, x] = mag[y, x]
         else:
            nonmax[y, x] = 0
   ##########################################################################
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
     theta  - a 2D numpy array, containing edge direction in [0, 2*pi)

   Returns:
     edge   - a 2D numpy array, containing edges map where the edge pixel is 1 and 0 otherwise.
"""

def hysteresis_edge_linking(nonmax, theta):
   ##########################################################################
   # remove 0 values for threshold calculation
   nonmax_no0 = nonmax[nonmax != 0]
   high_thresh = min(np.percentile(nonmax_no0, 86.5), np.max(nonmax) * 0.25)
   low_thresh = 0.5 * high_thresh
   img_r, img_c = nonmax.shape
   edge = np.zeros((img_r, img_c))
   strong_edges_y, strong_edges_x = np.where(nonmax >= high_thresh)
   strong_edges_y = list(strong_edges_y)
   strong_edges_x = list(strong_edges_x)

   while len(strong_edges_y) > 0:
      y, x = strong_edges_y.pop(), strong_edges_x.pop()
      edge[y, x] = 1 # set strong edge to 1 

      # get edge direction
      if abs(np.sin(theta[y,x])) < np.sin(np.pi / 8):
         ox, oy = 1, 0 # horizontal edge
      elif abs(np.sin(theta[y,x])) > np.sin(3 * np.pi / 8):
         ox, oy = 0, 1 # vertical edge
      elif np.cos(theta[y,x]) > 0:
         ox, oy = 1, 1  # diagonal edge in pi/4 direction
      else:
         ox, oy = -1, 1 # diagonal edge in 3 pi/4 direction

      # check neighbors for weak edges
      if 0 <= y + oy < img_r and 0 <= x + ox < img_c and \
         low_thresh < nonmax[y + oy, x + ox] < high_thresh and edge[y + oy, x + ox] == 0:

         edge[y + oy, x + ox] = 1 # declare weak edge as strong
         strong_edges_y.append(y + oy)
         strong_edges_x.append(x + ox)
      elif 0 <= y - oy < img_r and 0 <= x - ox < img_c and \
         low_thresh < nonmax[y - oy, x - ox] < high_thresh and edge[y - oy, x - ox] == 0:

         edge[y - oy, x - ox] = 1 # declare weak edge as strong 
         strong_edges_y.append(y - oy)
         strong_edges_x.append(x - ox)

   ##########################################################################
   return edge


def estimate_edge_strength_direction(dx, dy):
   # Takes dx and dy outputs from sobel_gradients() and computes edge
   # strength and direction at each pixel
   assert(dx.shape == dy.shape)
   edge_strengths = np.zeros(dx.shape)
   edge_angles = np.zeros(dx.shape)

   stacked_gradients = np.dstack((dx, dy))
   edge_strengths = linalg.norm(stacked_gradients, axis = 2)
   gradient_angles = np.arctan2(dy, dx) # output in range [-pi, pi]
   # edge is orthogonal to gradient 
   edge_angles = (gradient_angles % np.pi + (np.pi / 2)) % np.pi # range is [0, pi]
   
   return edge_strengths, edge_angles

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
def canny(image, downsample_factors = [1]):
   ##########################################################################
   all_edge_strengths = []

   for factor in downsample_factors:
      downsampled_image = smooth_and_downsample(image, factor)
      # add some smoothing when calculating the sobel gradients
      dx, dy = sobel_gradients(downsampled_image)
      edge_strengths_d, edge_angles_d = estimate_edge_strength_direction(dx, dy)
      edge_strengths = bilinear_upsampling(edge_strengths_d, factor)

      if all_edge_strengths == []:
         all_edge_strengths = edge_strengths.reshape(edge_strengths.shape + (1,))
      else:
         all_edge_strengths = np.dstack((edge_strengths, all_edge_strengths))

   # It is better to use the angles from the original image because 
   # averaging angles is incorrect when you have them at opposite ends of the range 
   denoised_image = denoise_gaussian(image, sigma = 0.5)
   dx, dy = sobel_gradients(denoised_image)
   no_use, edge_angles_use = estimate_edge_strength_direction(dx, dy)

   # estimate edge strength and direction
   mag = np.mean(all_edge_strengths, axis = 2)

   # perform nomax suppression
   nonmax = nonmax_suppress(mag, edge_angles_use)

   # edge map after edge linking
   edge = hysteresis_edge_linking(nonmax, edge_angles_use)
   ##########################################################################
   return mag, nonmax, edge