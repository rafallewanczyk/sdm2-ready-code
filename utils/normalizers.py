import numpy as np

def skeleton_normalizer(single_skeleton):
    x_offset = np.min(single_skeleton[:, :, 0])
    y_offset = np.min(single_skeleton[:, :, 1])
    frame_array = single_skeleton[:, ] - (x_offset, y_offset)

    w = np.ceil(np.max(frame_array[:, :, 0]))
    h = np.ceil(np.max(frame_array[:, :, 1]))

    normalized = frame_array[:, ] / (w, h)
    return normalized
