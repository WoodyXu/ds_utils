# compute the Intersection over Union value.

def iou(a, b):
    """
    Parameters:
        a: bounding box, (x1, y1, x2, y2)
        b: bounding box, (x1, y1, x2, y2)
    Returns:
        intersection over union
    """
    def intersection(a, b):
        left_x = max(a[0], b[0])
        bottom_y = max(a[1], b[1])
        right_x = min(a[2], b[2])
        up_y = min(a[3], b[3])

        w = right_x - left_x
        h = up_y - bottom_y

        if w < 0 or h < 0:
            return 0
        else:
            return w * h

    def union(a, b, common_area):
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return area_a + area_b - common_area

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    intersection_area = intersection(a, b)
    union_area = union(a, b, intersection_area)

    return intersection_area * 1.0 / (union_area + 1e-6)
