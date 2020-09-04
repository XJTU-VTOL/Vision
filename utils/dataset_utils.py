import numpy as np

def calculate_corners(bbox):
    """
        bbox: numpy array (N, 4) bbox representations in center_x, center_y, width, height

        return :
            numpy array (N, 4, 2):
                left_top_x, left_top_y, 
                right_top_x, right_top_y, 
                left_bottom_x, left_bottom_y, 
                right_bottom_x, right_bottom_y 
    """
    left_top = bbox[:, [0, 1]] - bbox[:, [2, 3]] / 2
    right_top = np.stack([bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] - bbox[:, 3] / 2], axis=1)
    left_bottom = np.stack([bbox[:, 0] - bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3] / 2], axis=1)
    right_bottom = bbox[:, [0, 1]] + bbox[:, [2, 3]] / 2

    return np.stack([left_top, right_top, left_bottom, right_bottom], axis=1)