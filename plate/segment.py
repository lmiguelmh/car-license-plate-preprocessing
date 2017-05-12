"""
@author lmiguelmh
@since 20170416
"""

import cv2
import numpy as np


def segment_plates(mat, coordinates, only_first=True):
    plates = []
    for coordinate in coordinates:
        # actual corners
        src_corners = np.full((4, 2), 0, dtype=np.float32)
        topl = src_corners[0] = coordinate[0]
        topr = src_corners[1] = coordinate[1]
        bottomr = src_corners[2] = coordinate[2]
        bottoml = src_corners[3] = coordinate[3]

        # topl = coordinate[0]
        # topr = coordinate[1]
        # bottomr = coordinate[2]
        # bottoml = coordinate[3]

        # calc the maxwidth & maxheight for affine transformation
        bottom_width = np.sqrt(((bottomr[0] - bottoml[0]) ** 2) + ((bottomr[1] - bottoml[1]) ** 2))
        top_width = np.sqrt(((topr[0] - topl[0]) ** 2) + ((topr[1] - topl[1]) ** 2))
        right_height = np.sqrt(((topr[0] - bottomr[0]) ** 2) + ((topr[1] - bottomr[1]) ** 2))
        left_height = np.sqrt(((topl[0] - bottoml[0]) ** 2) + ((topl[1] - bottoml[1]) ** 2))
        max_width = max(int(bottom_width), int(top_width))
        max_height = max(int(right_height), int(left_height))

        # corners after transformation
        # http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getperspectivetransform
        dst_corners = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
                               dtype="float32")
        transformation_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
        roi = cv2.warpPerspective(mat, transformation_matrix, (max_width, max_height))
        plates.append(roi)

        if only_first:
            break

    return plates


def draw_segmentation_mask(w, h, contours, selected):
    bin_cnt = np.zeros((h, w), np.uint8)
    for i, cnt in enumerate(contours):
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if i in selected:
            rxf = rx + rw - 1
            ryf = ry + rh - 1
            cv2.rectangle(bin_cnt, (rx, ry), (rxf, ryf), 255, -1)
    return bin_cnt


def process_mask(gray, rectangles_mask, min_contour_area=20):
    h, w = gray.shape
    final = np.zeros((h, w), np.uint8)
    final = cv2.bitwise_not(final)
    image, contours, hierarchy = cv2.findContours(rectangles_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        rxf = rx + rw - 1
        ryf = ry + rh - 1

        _, character = cv2.threshold(gray[ry:ryf, rx:rxf], 0, 255, cv2.THRESH_OTSU)
        # print("character")
        # _, character_contours, _ = cv2.findContours(character, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # for _, character_cnt in enumerate(character_contours):
        #     contour_area = cv2.contourArea(character_cnt)
        #     print(contour_area)
        #     if contour_area > min_contour_area:
        #         final[ry:ryf, rx:rxf] = character
        final[ry:ryf, rx:rxf] = character
    return final



def clean_img_bin(img_bin, min_area, max_area):
    h, w = img_bin.shape
    _, character_contours, _ = cv2.findContours(img_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    for i, character_cnt in enumerate(character_contours):
        contour_area = cv2.contourArea(character_cnt)
        if min_area < contour_area < max_area:
            cv2.drawContours(mask, character_contours, i, (255, 255, 255), cv2.FILLED, 8)
    final2 = cv2.bitwise_and(img_bin, img_bin, mask=mask)
    bk = cv2.bitwise_not(np.zeros((h, w), np.uint8))
    final2_bk = cv2.bitwise_and(bk, bk, mask=cv2.bitwise_not(mask))
    return cv2.bitwise_or(final2, final2_bk)
