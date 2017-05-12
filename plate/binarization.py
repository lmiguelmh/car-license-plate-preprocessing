"""
@author lmiguelmh
@since 20170505
"""
import cv2
import numpy as np


# from http://www.learnopencv.com/
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def laplacian_adaptive_thresholding(gray):
    h, w = gray.shape
    window_size = ((h >> 1) << 1) + 1
    gx = int(window_size / 5)
    gx = gx + (1 - gx % 2)
    gy = int(window_size / 5)
    gy = gy + (1 - gy % 2)
    filtered_blurred = cv2.GaussianBlur(gray, (gx, gy), 0)
    img_back_gray_filtered = cv2.Laplacian(filtered_blurred, cv2.CV_64F)
    img_back_gray_filtered = cv2.normalize(img_back_gray_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return cv2.adaptiveThreshold(img_back_gray_filtered, 255, cv2.THRESH_BINARY_INV,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, window_size, 0)
