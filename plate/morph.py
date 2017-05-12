"""
@author lmiguelmh
@since 20170510
"""
import cv2
import numpy as np
import iou


def clean_contours(img_bin, border_radius=10, min_iou_ratio=0.005, max_iou_ratio=0.75):
    h, w = img_bin.shape

    # add a "border" so a char thats in the borders is not detected like a outer contour
    img_bin[0, :] = 255
    img_bin[h - 1, :] = 255
    img_bin[:, 0] = 255
    img_bin[:, w - 1] = 255

    image, contours, hierarchy = cv2.findContours(img_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    maxrect = [0, 0, w, h]
    selected_contours = []
    for i, cnt in enumerate(contours):
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        rxf = rx + rw - 1
        ryf = ry + rh - 1
        iou_ratio = iou.bb_intersection_over_union(maxrect, [rx, ry, rxf, ryf])
        has_right_size = min_iou_ratio < iou_ratio and iou_ratio < max_iou_ratio
        close_to_border = rxf < border_radius or ryf < border_radius or rx + border_radius > w or ry + border_radius > h
        if has_right_size and not close_to_border:
            selected_contours.append(i)
    return contours, selected_contours


# characters in black, background in white
def clean_img_bin(img_bin, min_area, max_area, border_radius=10):
    h, w = img_bin.shape

    # add a "border" so a char thats in the borders is not detected like a outer contour
    img_bin[0, :] = 255
    img_bin[h - 1, :] = 255
    img_bin[:, 0] = 255
    img_bin[:, w - 1] = 255

    _, character_contours, _ = cv2.findContours(img_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    for i, character_cnt in enumerate(character_contours):
        rx, ry, rw, rh = cv2.boundingRect(character_cnt)
        rxf = rx + rw - 1
        ryf = ry + rh - 1
        contour_area = cv2.contourArea(character_cnt)
        has_right_area = min_area < contour_area < max_area
        close_to_border = rxf < border_radius or ryf < border_radius or rx + border_radius > w or ry + border_radius > h
        if has_right_area and not close_to_border:
            cv2.drawContours(mask, character_contours, i, (255, 255, 255), cv2.FILLED, 8)

    final2 = cv2.bitwise_and(img_bin, img_bin, mask=mask)
    bk = cv2.bitwise_not(np.zeros((h, w), np.uint8))
    final2_bk = cv2.bitwise_and(bk, bk, mask=cv2.bitwise_not(mask))
    return cv2.bitwise_or(final2, final2_bk)


# def draw_segment_mask(img_bin, border_radius=5, min_iou_ratio=0.005, max_iou_ratio=0.75):
#     h, w = img_bin.shape
#     image, contours, hierarchy = cv2.findContours(img_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#     bin_cnt = np.zeros(img_bin.shape, np.uint8)
#     maxrect = [0, 0, w, h]
#     border_radius = 5
#     for i, cnt in enumerate(contours):
#         rx, ry, rw, rh = cv2.boundingRect(cnt)
#         rxf = rx + rw - 1
#         ryf = ry + rh - 1
#         iou_ratio = iou.bb_intersection_over_union(maxrect, [rx, ry, rxf, ryf])
#         has_right_size = iou_ratio > 0.005 and iou_ratio < 0.75
#         close_to_border = rxf < border_radius or ryf < border_radius or rx + border_radius > w or ry + border_radius > h
#         if has_right_size and not close_to_border:
#             cv2.rectangle(bin_cnt, (rx, ry), (rxf, ryf), 255, -1)