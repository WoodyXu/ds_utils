# Region Proposal Network

def calc_rpn_label(img_data, width, height, resized_width, resize_height, downsampling_ratio,
        anchor_box_scales, anchor_box_ratios):
    """
    calculate rpn ground truth for one image
    Parameters:
        img_data: a dict
            {"filepath": /data/000000.png, "width": 224, "height": 224,
             "bboxes": [{"class": "car", "x1":1.0, "x2": 1.5, "y1": 1.0, "y2":1.5}, ...]}
        width: original image width
        height: original image height
        resized_width: resized width
        resized_height: resized height
        downsampling_ratio: input size / feature map size
        anchor_box_scales: a list of sizes of the anchors
        anchor_box_ratios: a list of aspect ratios of the anchors
    """
    num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)

