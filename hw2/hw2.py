import numpy as np
from hw1 import *

"""
   INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)

   Implement an interest point operator of your choice.

   Your operator could be:

   (A) The Harris corner detector (Szeliski 7.1.1)

               OR

   (B) The Difference-of-Gaussians (DoG) operator defined in:
       Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
       https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

               OR

   (C) Any of the alternative interest point operators appearing in
       publications referenced in Szeliski or in lecture

              OR

   (D) A custom operator of your own design

   You implementation should return locations of the interest points in the
   form of (x,y) pixel coordinates, as well as a real-valued score for each
   interest point.  Greater scores indicate a stronger detector response.

   In addition, be sure to apply some form of spatial non-maximum suppression
   prior to returning interest points.

   Whichever of these options you choose, there is flexibility in the exact
   implementation, notably in regard to:

   (1) Scale

       At what scale (e.g. over what size of local patch) do you operate?

       You may optionally vary this according to an input scale argument.

       We will test your implementation at the default scale = 1.0, so you
       should make a reasonable choice for how to translate scale value 1.0
       into a size measured in pixels.

   (2) Nonmaximum suppression

       What strategy do you use for nonmaximum suppression?

       A simple (and sufficient) choice is to apply nonmaximum suppression
       over a local region.  In this case, over how large of a local region do
       you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.

   If you detect more interest points than the requested maximum (given by
   the max_points argument), return only the max_points highest scoring ones.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy array
      max_points  - maximum number of interest points to return
      scale       - (optional, for your use only) scale factor at which to
                    detect interest points
      mask        - (optional, for your use only) foreground mask constraining
                    the regions to extract interest points
   Returns:
      xs          - numpy array of shape (N,) containing x-coordinates of the
                    N detected interest points (N <= max_points)
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing a real-valued
                    measurement of the relative strength of each interest point
                    (e.g. corner detector criterion OR DoG operator magnitude)
"""
def find_interest_points(image, max_points = 200, scale = 1.0, mask = None):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'

   # set up helper data structures 
   thresh = 0
   cand = []
   row, col = image.shape
   candGrid = np.zeros((row, col))
   candGrid2 = np.zeros((row,col))
   s = int(np.ceil(scale))   
   

   # set up denoising variables
   dx, dy = sobel_gradients(image)
   dxx = np.multiply(dx, dx)
   dxy = np.multiply(dx, dy)
   dyy = np.multiply(dy, dy)
   dXX = denoise_gaussian(dxx,sigma=scale)
   dXY = denoise_gaussian(dxy,sigma=scale)
   dYY = denoise_gaussian(dyy,sigma=scale)


   # calculate and save nonzero R vals via Harris Point Operator cornerness fn
   for y in range(row):
      for x in range(col):
         lx = dXX[y, x]
         lxy = np.square(dXY[y, x])
         ly = dYY[y, x]
         R = (lx * ly) - lxy - 0.05 * (lx + ly) * (lx + ly)
         # ignore low R vals
         if R <= 0:
            continue
         candGrid[y, x] = R
         cand.append((int(x),int(y)))
   
   # apply mask if mask and zero out relevant cords
   if mask is not None: 
      non_zero_indices = mask == 0
      candGrid[non_zero_indices] = 0

   # prune non-local maximum 
   for (x,y) in cand:
      R = candGrid[y, x]
      for i in range(max(0,y - s),min(row,y + s + 1)):
         for j in range(max(0,x - s),min(col,x + s + 1)):
            # set R to 0 if non local max
            if candGrid[y,x] < candGrid[i,j]:
               R = 0

      candGrid2[y, x] = R

   
   # count current non zero points
   currPoints = 0
   for i in range(row):
      for j in range(col):
         if candGrid2[i,j] != 0:
                  currPoints += 1

   
   # apply max threshholding
   if currPoints > max_points:
      pV = 100 - (100 * max_points / currPoints)
      coords = candGrid2[candGrid2 != 0]
      thresh = np.percentile(coords, pV)

   xs,ys,scores = [], [], []

   for i in range(len(candGrid2)):
      for j in range(len(candGrid2[i])):
         if candGrid2[i][j] > thresh:
            xs = np.append(xs, j) 
            ys = np.append(ys, i) 
            scores = np.append(scores, candGrid2[i][j]) 

   return np.array(xs), np.array(ys), np.array(scores)

"""
   FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)

   Implement a SIFT-like feature descriptor by binning orientation energy
   in spatial cells surrounding an interest point.

   Unlike SIFT, you do not need to build-in rotation or scale invariance.

   A reasonable default design is to consider a 3 x 3 spatial grid consisting
   of cell of a set width (see below) surrounding an interest point, marked
   by () in the diagram below.  Using 8 orientation bins, spaced evenly in
   [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions.

             ____ ____ ____
            |    |    |    |
            |    |    |    |
            |____|____|____|
            |    |    |    |
            |    | () |    |
            |____|____|____|
            |    |    |    |
            |    |    |    |
            |____|____|____|

                 |----|
                  width

   You will need to decide on a default spatial width.  Optionally, this can
   be a multiple of a scale factor, passed as an argument.  We will only test
   your code by calling it with scale = 1.0.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

  Arguments:
      image    - a grayscale image in the form of a 2D numpy
      xs       - numpy array of shape (N,) containing x-coordinates
      ys       - numpy array of shape (N,) containing y-coordinates
      scale    - scale factor

   Returns:
      feats    - a numpy array of shape (N,K), containing K-dimensional
                 feature descriptors at each of the N input locations
                 (using the default scheme suggested above, K = 72)
"""
def extract_features(image, xs, ys, scale = 1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'

   # set up variables
   out = np.zeros((len(xs) , 72))
   cw = int(scale) * 3
   p = cw + (cw // 2)
   image = mirror_border(image, wx = p, wy = p)
   rows,cols = image.shape
   mag = np.zeros((rows, cols))
   index = 0
   pts = zip(list(map(int, xs)),list(map(int, ys)))

   # get mag and theta grids
   dx, dy = sobel_gradients(denoise_gaussian(image, sigma = scale))
   for i in range(rows):
      for j in range(cols):
         mag[i, j] = np.sqrt(dx[i, j] ** 2 + dy[i, j] ** 2)
   diR = np.arctan2(dy, dx)

   for (x,y) in pts:
         hist = np.array([])
         # grab dir and mag windows around x,y
         fac = 2 * p + 1
         wT = diR[y: y + fac, x: x + fac]
         wM = mag[y: y + fac, x: x + fac]
         wtR = [wT[i:i + 3, :] for i in range(0, 9, 3)]
         wmR = [wM[i:i + 3, :] for i in range(0, 9, 3)]

         # iterate through each instance in window
         for i in range(3):
            for j in range(3):
               cT = np.hsplit(wtR[i], 3)[j]
               cMag = np.hsplit(wmR[i],3)[j]
               grad_sum = np.sum(cMag)
               # handle high gradients
               for ii in range(3):
                     for jj in range(3):
                        if cMag[ii][jj] > grad_sum * 0.2:
                           cMag[ii][jj] = grad_sum * 0.2    
               h, _ = np.histogram(cT, bins = 
                                 np.linspace(-np.pi, np.pi, 9), 
                                 weights = cMag, density = True)
               hist = np.append(hist, h)
         out[index,:] = hist
         index += 1
   return out


"""
   FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)

   Given two sets of feature descriptors, extracted from two different images,
   compute the best matching feature in the second set for each feature in the
   first set.

   Matching need not be (and generally will not be) one-to-one or symmetric.
   Calling this function with the order of the feature sets swapped may
   result in different returned correspondences.

   For each match, also return a real-valued score indicating the quality of
   the match.  This score could be based on a distance ratio test, in order
   to quantify distinctiveness of the closest match in relation to the second
   closest match.  It could optionally also incorporate scores of the interest
   points at which the matched features were extracted.  You are free to
   design your own criterion.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                 feature descriptors (generated via extract_features())
      feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                 feature descriptors (generated via extract_features())
      scores0  - a numpy array of shape (N0,) containing the scores for the
                 interest point locations at which feats0 was extracted
                 (generated via find_interest_point())
      scores1  - a numpy array of shape (N1,) containing the scores for the
                 interest point locations at which feats1 was extracted
                 (generated via find_interest_point())

   Returns:
      matches  - a numpy array of shape (N0,) containing, for each feature
                 in feats0, the index of the best matching feature in feats1
      scores   - a numpy array of shape (N0,) containing a real-valued score
                 for each match
"""
def match_features(feats0, feats1, scores0, scores1):
    fN = feats0.shape[0]
    matches = np.zeros(fN, dtype=int)
    scores = np.zeros(fN)

    for i in range(fN):
        cF = feats0[i, :]
        d_vec = feats1 - cF
        d_norm = np.linalg.norm(d_vec, axis=1)
        sortnorm = np.argsort(d_norm)

        iFirst = sortnorm[0]
        iSecond = sortnorm[1]

        if (d_norm[iFirst] != 0) and (d_norm[iSecond] !=0):
            ratio = d_norm[iFirst] / d_norm[iSecond]
            sc = 1 / ratio
        else:
            # set scalar to 1.0 if either distance norm is 0
            sc = 1.0

        matches[i] = int(iFirst)
        scores[i] = sc * (scores0[i] + scores1[iFirst])

    return matches, scores

"""
   HOUGH TRANSFORM (7 Points Implementation + 3 Points Write-up)

   Assuming two images of the same scene are related primarily by
   translational motion, use a predicted feature correspondence to
   estimate the overall translation vector t = [tx ty].

   Your implementation should use a Hough transform that tallies votes for
   translation parameters.  Each pair of matched features votes with some
   weight dependant on the confidence of the match; you may want to use your
   estimated scores to determine the weight.

   In order to accumulate votes, you will need to decide how to discretize the
   translation parameter space into bins.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      xs0     - numpy array of shape (N0,) containing x-coordinates of the
                interest points for features in the first image
      ys0     - numpy array of shape (N0,) containing y-coordinates of the
                interest points for features in the first image
      xs1     - numpy array of shape (N1,) containing x-coordinates of the
                interest points for features in the second image
      ys1     - numpy array of shape (N1,) containing y-coordinates of the
                interest points for features in the second image
      matches - a numpy array of shape (N0,) containing, for each feature in
                the first image, the index of the best match in the second
      scores  - a numpy array of shape (N0,) containing a real-valued score
                for each pair of matched features

   Returns:
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
      max_votes - the max value in the matrix storing vote tallies;
                  this output is provided for your own convenience and 
                  you are free to design its format
"""
def hough_votes(xs0, ys0, xs1, ys1, matches, scores):
   discreteP = 30 
   v = np.zeros((discreteP, discreteP))
   N = len(matches)
   tx,ty = np.zeros(N), np.zeros(N)
   txMi, tyMi = np.inf, np.inf
   txMX, tyMX = 0,0

   # calculate difference vectos and get widest dimensions for line space calc
   for i in range(N):
      ty[i] = ys1[matches[i]] - ys0[i]
      tx[i] = xs1[matches[i]] - xs0[i]
      if tx[i] < txMi:
         txMi = tx[i]
      if tx[i] > txMX:         
         txMX = tx[i]
      if ty[i] < tyMi:
         tyMi = ty[i]
      if ty[i] > tyMX:         
         tyMX = ty[i]

   txE, _ = np.linspace(txMi, txMX, discreteP, endpoint = False, retstep = True) 
   tyE, _ = np.linspace(tyMi, tyMX, discreteP, endpoint = False, retstep = True)

   tx_out, ty_out, max_value = 0, 0, 0

   for i in range(N):
      xI, yI = 0, 0
      cy, cx = ty[i],tx[i]
      for j in range(discreteP):
         # save best bin for a given i based on max score in tx, ty
         if cx >= txE[j]:
            xI = j
         if cy >= tyE[j]:
            yI = j
      v[yI, xI] += scores[i]

      if v[yI, xI] > max_value:
            max_value = v[yI, xI]
            tx_out, ty_out = txE[xI], tyE[yI]
   
   return tx_out, ty_out, max_value

"""
    OBJECT DETECTION (10 Points Implementation + 5 Points Write-up)

    Implement an object detection system which, given multiple object
    templates, localizes the object in the input (test) image by feature
    matching and hough voting.

    The first step is to match features between template images and test image.
    To prevent noisy matching from background, the template features should
    only be extracted from foreground regions.  The dense point-wise matching
    is then used to compute a bounding box by hough voting, where box center is
    derived from voting output and the box shape is simply the size of the
    template image.

    To detect potential objects with diversified shapes and scales, we provide
    multiple templates as input.  To further improve the performance and
    robustness, you are also REQUIRED to implement a multi-scale strategy
    either:
       (a) Implement multi-scale interest points and feature descriptors OR
       (b) Repeat a single-scale detection procedure over multiple image scales
           by resizing images.

    In addition to your implementation, include a brief write-up (in hw2.pdf)
    of your design choices on multi-scale implementaion and samples of
    detection results (please refer to display_bbox() function in visualize.py).

    Arguments:
        template_images - a list of gray scale images.  Each image is in the
                          form of a 2d numpy array which is cropped to tightly
                          cover the object.

        template_masks  - a list of binary masks having the same shape as the
                          template_image.  Each mask is in the form of 2d numpy
                          array specyfing the foreground mask of object in the
                          corresponding template image.

        test_img        - a gray scale test image in the form of 2d numpy array
                          containing the object category of interest.

    Returns:
         bbox           - a numpy array of shape (4,) specifying the detected
                          bounding box in the format of
                             (x_min, y_min, x_max, y_max)

"""
def object_detection(template_images, template_masks, test_img):
   tIM = zip(template_images, template_masks)
   # extract test image feats 
   xs1, ys1, test_scores = find_interest_points(test_img, max_points = 400, scale = 1)
   feats1 = extract_features(test_img, xs1, ys1, scale = 1)
   rows, cols = test_img.shape

   # multi-scale intrest points and features, must be odd for gaussian sigma
   sF =  (1, 3, 5) 
   # sF = [1] # for single scale implimentation
   maxv,xM,yM,xS,yS = 0, 0, 0, 0, 0

   for (tM, tMM) in tIM:
      for s in sF:
         # run object detection approach on given scale 
         xs, ys, scores = find_interest_points(tM, scale = s, max_points = 400, mask = tMM)
         feats = extract_features(tM, xs, ys, scale = s)
         matches, match_scores = match_features(feats, feats1, scores, test_scores)
         tx, ty, votes = hough_votes(xs, ys, xs1, ys1, matches, match_scores)
         # save best performaning scale feature pair shape, votes, and dimensions
         if votes > maxv:
            maxv = votes
            xM,yM = tx,ty
            yS,xS = tM.shape

   # get box dimensions
   xM = max(0, min(xM, cols - 1))
   xMM = max(0, min(xM + xS, cols - 1))
   yM = max(0, min(yM, rows - 1))
   yMM = max(0, min(yM + yS, rows - 1))
   return [xM, yM, xMM, yMM]