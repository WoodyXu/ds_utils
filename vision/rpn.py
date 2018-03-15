# Region Proposal Network
import numpy as np
import random

import iou

def calc_rpn_label_regr(img_data, width, height, resized_width, resize_height, downsampling_ratio,
        anchor_box_sizes, anchor_box_ratios, rpn_max_overlap, rpn_min_overlap):
    """
    calculate rpn ground truth for one image
    Parameters:
        img_data: a dict
            {"filepath": /data/000000.png, "width": 224, "height": 224,
             "bboxes": [ {"class": "car", "x1": 1.0, "x2": 1.5, "y1": 1.0, "y2": 1.5}, ... ]
            }
        width: original image width
        height: original image height
        resized_width: resized width as model input
        resized_height: resized height as model input
        downsampling_ratio: input size / feature map size
        anchor_box_sizes: a list of sizes of the anchors
        anchor_box_ratios: a list of aspect ratios of the anchors
        rpn_max_overlap, rpn_min_overlap: the lower and upper threshold for the iou of rpn
    Returns:
        (rpn_labels, rpn_regr)
        rpn_labels: (1, feature_map_width, feature_map_height, 2 * number_anchors)
        rpn_regr: (1, feature_map_width, featutre_map_height, 2 * 4 * number_anchors)
    """
    num_anchors = len(anchor_box_sizes) * len(anchor_box_ratios)
    num_anchor_ratios = len(anchor_box_ratios)
    num_bboxes = len(img_data["bboxes"])
    # get the output feature map size based on the model architecture downsampling ratio
    fm_width, fm_height = resized_width / downsampling_ratio, resize_height / downsampling_ratio

    # stores the label of each anchor, indicating whether this anchor contains an object or not
    y_rpn_label = np.zeros((fm_height, fm_width, num_anchors))
    # stores the validness of each anchor, indicating whether this anchor has a label or not
    y_is_box_valid = np.zeros((fm_height, fm_width, num_anchors))
    # stores the delta regressions of each anchor,
    # [delta_center_x, delta_center_y, delta_width, delta_height]
    y_rpn_regr = np.zeros((fm_height, fm_width, num_anchors * 4))

    # number of anchors that one bounding box contains
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    # the best anchor that one bounding box contains
    # [ feature_map_row_pixel_index, feature_map_column_pixel_index, anchor_ratio_index, anchor_size_index ]
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    # the best iou that one bounding box intersects with anchors
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    # the best anchor regression target that one bounding box contains
    # [ delta_center_x, delta_center_y, delta_width, delta_height ]
    best_delta_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # convert bounding boxes in original images to that in resized images
    # columns: [ x1, x2, y1, y2 ]
    gta = np.zeros((num_bboxes, 4))
    for index, bbox in enumerate(img_data["bboxes"]):
        gta[index, 0] = bbox["x1"] * (resized_width * 1.0 / width)
        gta[index, 1] = bbox["x2"] * (resized_width * 1.0 / width)
        gta[index, 2] = bbox["y1"] * (resized_height * 1.0 / height)
        gta[index, 3] = bbox["y2"] * (resized_height * 1.0 / height)

    # we start to iterate each combination of anchors
    for anchor_size_idx in range(len(anchor_box_sizes)):
        for anchor_ratio_idx in range(num_anchor_ratios):
            # first we determine the (width, height) of the anchor
            anchor_width = anchor_box_sizes[anchor_size_idx] * anchor_box_ratios[anchor_ratio_idx][0]
            anchor_height = anchor_box_sizes[anchor_size_idx] * anchor_box_ratios[anchor_ratio_idx][1]

            # then we traverse the feature map plane
            for ix in range(fm_width):
                # the anchor coordinates in resized image input
                anchor_x1 = downsampling_ratio * (ix + 0.5) - anchor_width / 2
                anchor_x2 = downsampling_ratio * (ix + 0.5) + anchor_width / 2

                if anchor_x1 < 0 or anchor_x2 > resized_width:
                    continue

                for jy in range(fm_height):
                    # the anchor coordinates in resized image input
                    anchor_y1 = downsampling_ratio * (yj + 0.5) - anchor_height / 2
                    anchor_y2 = downsampling_ratio * (yj + 0.5) + anchor_height / 2

                    if anchor_y1 < 0 or anchor_y2 > resized_height:
                        continue

                    # ok, until now we get the specific anchor in resized image 
                    # (anchor_x1, anchor_x2, anchor_y1, anchor_y2)
                    current_anchor_coord = [ anchor_x1, anchor_y1, anchor_x2, anchor_y2 ]

                    anchor_label = "neg"
                    best_iou_for_anchor = 0.0

                    for bbox_idx in range(num_bboxes):
                        current_bbox_coord = [ gta[bbox_idx, 0], gta[bbox_idx, 2], gta[bbox_idx, 1], gta[bbox_idx, 3] ]
                        current_iou = iou.iou(current_bbox_coord, current_anchor_coord)

                        # calculate regression target
                        center_bbox_x = (gta[bbox_idx, 0] + gta[bbox_idx, 1]) / 2.0
                        center_bbox_y = (gta[bbox_idx, 2] + gta[bbox_idx, 3]) / 2.0
                        center_anchor_x = (anchor_x1 + anchor_x2) / 2.0
                        center_anchor_y = (anchor_y1 + anchor_y2) / 2.0
                        bbox_width = gta[bbox_idx, 1] - gta[bbox_idx, 0]
                        bbox_height = gta[bbox_idx, 3] - gta[bbox_idx, 2]

                        delta_center_x = (center_bbox_x - center_anchor_x) / anchor_width
                        delta_center_y = (center_bbox_y - center_anchor_y) / anchor_height
                        delta_width = np.log(bbox_width / anchor_width)
                        delta_height = np.log(bbox_height / anchor_height)

                        # we should help non-background bounding box find a best anchor
                        if img_data["bboxes"][bbox_idx]["class"] != "bg":
                            if current_iou > best_iou_for_bbox[bbox_idx]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = current_ious
                                best_delta_for_bbox[bbox_num, :] = [delta_center_x, delta_center_y, delta_width, delta_height]

                        # if the current iou surpasses the upper threshold, we will set the anchor
                        # label to be true
                        if current_iou > rpn_max_overlap:
                            anchor_label = "pos"
                            num_anchors_for_bbox[bbox_num] += 1
                            # we should find the best regression target
                            if current_iou > best_iou_for_anchor:
                                best_iou_for_anchor = current_iou
                                best_regr = (delta_center_x, delta_center_y, delta_width, delta_height)

                        # if the current iou is in between lower and upper threshold, we will not
                        # set the anchor label
                        if current_ious > rpn_min_overlap and current_ious < rpn_max_overlap:
                            if anchor_label != "pos":
                                anchor_label = "neutral"

                    # determine the classification target
                    if anchor_label == "neg":
                        y_is_box_valid[jy, ix, num_anchor_ratios * anchor_size_idx + anchor_ratio_idx] = 1
                        y_rpn_label[jy, ix, num_anchor_ratios * anchor_size_idx + anchor_ratio_idx] = 0
                    elif anchor_label == "neutral":
                        y_is_box_valid[jy, ix, num_anchor_ratios * anchor_size_idx + anchor_ratio_idx] = 0
                        y_rpn_label[jy, ix, num_anchor_ratios * anchor_size_idx + anchor_ratio_idx] = 0
                    elif anchor_label == "pos":
                        y_is_box_valid[jy, ix, num_anchor_ratios * anchor_size_idx + anchor_ratio_idx] = 1
                        y_rpn_label[jy, ix, num_anchor_ratios * anchor_size_idx + anchor_ratio_idx] = 1
                        start = 4 * (num_anchor_ratios * anchor_size_idx + anchor_ratio_idx)
                        y_rpn_regr[jy, ix, start: start + 4] = best_regr


    # maybe some ground truth bounding box has no anchors iou more than upper threshold,
    # we should assign the best anchor for the ground truth
    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            jy, ix, ratio_index, size_index = best_anchor_for_bbox[idx, :]
            y_is_box_valid[jy, ix, num_anchor_ratios * size_index + ratio_index] = 1
            y_rpn_label[jy, ix, num_anchor_ratios * size_index + ratio_index] = 1
            start = 4 * (num_anchor_ratios * size_index + ratio_index)
            y_rpn_regp[jy, ix, start: start + 4] = best_delta_for_bbox[idx, :]

    y_rpn_label = np.expand_dims(y_rpn_label, axis=0)
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    positives = np.where(np.logical_and(y_rpn_label[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    negatives = np.where(np.logical_and(y_rpn_label[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_positives = len(positives[0])
    num_negatives = len(negatives[0])

    # normally the rpn has more negatives than positives, so we close some negatives, and limit the
    # total number
    num_regions = 256

    if num_positives > num_regions / 2:
        sampled_positives = random.sample(range(num_positives), num_positives - num_regions / 2)
        y_is_box_valid[0, positives[0][sampled_positives], positives[1][sampled_positives], positives[2][sampled_positives]] = 0
        num_positives = num_regions / 2

    if num_negatives + num_positives > num_regions:
        sampled_negatives = random.sample(range(num_negatives), num_negatives + num_positives - num_regions)
        y_is_box_valid[0, negatives[0][sampled_negatives], negatives[1][sampled_negatives], negatives[2][sampled_negatives]] = 0
        num_negatives = num_regions - num_positives

    # the result rpn classification labels, for the last axis, the first half part indicates whether
    # this anchor is a sample of not(contribute to the loss), the second half part indicates the
    # true labels
    result_rpn_labels = np.concatenate([y_is_box_valid, y_rpn_label], axis=3)
    # the result rpn regression targets, for the last axis, the first half part indicates whether the
    # (index + half length) postision should contribute to the regression loss, you know only the
    # anchors containing objects calculate the loss
    result_rpn_regr = np.concatenate([np.repeat(y_rpn_label, 4, axis=3), y_rpn_regr], axis=3)

    return np.copy(result_rpn_labels), np.copy(result_rpn_regr)
