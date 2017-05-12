"""
@author lmiguelmh
@since 20170512
"""

import os
import cv2
from plate import detect, segment, noise, roi, binarization, morph


def get_files(dir, ext):
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(ext)]


def process_plate(img_path, write_plate=True):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # plate points retrieving
    points = []
    points_path = img_path + '.pkz'
    if os.path.exists(points_path):
        points = roi.retrieve(points_path, decompress=False)
    else:
        print(img_path, ' points file for plate not found')
        return

    # plate segmentation
    plates = segment.segment_plates(img, [points])
    gray = cv2.cvtColor(plates[0], cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # bilateral filter
    wsize = h >> 3
    gray = cv2.bilateralFilter(gray, wsize, 30, wsize)

    # noise filtering
    filtered = noise.homomorphic(gray, 0.1, 1.)

    # binarization
    _, img_bin = cv2.threshold(filtered, 0, 255, cv2.THRESH_OTSU)
    # img_bin = cv2.dilate(img_bin, cv2.getStructuringElement(cv2.MORPH_ERODE, (2, 2)), iterations=1)

    # clean contours & dilate
    contours, selected = morph.clean_contours(img_bin)
    mask = segment.draw_segmentation_mask(w, h, contours, selected)
    # mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ERODE, (2, 2)), iterations=1)

    # contours #2 & segment
    final = segment.process_mask(filtered, mask)

    # contours #3
    final2 = morph.clean_img_bin(final, 20, h * w * 0.5)
    final2 = cv2.blur(final2, (2, 2), borderType=cv2.BORDER_REPLICATE)

    if write_plate:
        plate_path = img_path + "-plate.png"
        cv2.imwrite(plate_path, final2)
        print(plate_path, ' file written')

    return final2


def click_and_crop(event, x, y, flags, original_image_points):
    original, image, points = original_image_points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        if len(points) > 4:
            del points[0]

    image[:,:,:] = original[:,:,:]
    for point in points:
        cv2.circle(image, point, 4, color=(0, 255, 0), thickness=-1)