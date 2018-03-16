"""
show bounding boxes in detection or tracking
"""

import numpy as np
import cv2
import colorsys

def _create_color(tag, hue_step=0.41):
    """
    create an unique RGB color code for a specific tag
    """
    h, v = (tag * hue_step) % 1, 1.0 - (int(tag * hue_step) % 4) / 5
    r, g, b = colosys.hsv_to_rgb(h, 1.0, v)
    return int(255 * r), int(255 * g), int(255 * b)


def draw_bboxes_and_labels_on_image(img, class_mapping, bounding_boxes):
    """
    Parameters:
        img: image object by cv2.imread()
        class_mapping: number to name dict
            {1: "cat", 2: "mouse", ...}
        bounding_boxes: class number to boxes list
            {1: [box1, box2, ...], 2: [box3, box4, ...], ...}
            box: [x1, y1, x2, y2, prob]
    Returns:
        original image + boungding boxes + label texts 
    """
    for class_num, boxes in bounding_boxes.items():
        for box in boxes:
            assert len(box) == 5, "box must have five items: [x1, y1, x2, y2, prob]"
            x1, y1, x2, y2 = map(int, box[:4])
            prob = round(box[4], 2)

            unique_color = _create_color(class_num)
            text_label = "{} {}".format(class_mapping[class_num], prob)
            ret_val, base_line = cv2.getTextSize(test_label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)

            cv2.rectangle(img, (x1, y1), (x2, y2), unique_color, 2)
            cv2.rectangle(img, (x1 - 5, y1 + base_line - 5), (x1 + ret_val[0] + 5, y1 - ret_val[1] +
                5, unique_color, 2))
            cv2.rectangle(img, (x1 - 5, y1 + base_line - 5), (x1 + ret_val[0] + 5, y1 - ret_val[1] +
                5, unique_color, -1))
            cv2.putText(img, text_label, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    return img
