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
