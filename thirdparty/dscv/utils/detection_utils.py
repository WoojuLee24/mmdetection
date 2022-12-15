import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle


def pixel2inch(pixel):
    PIXEL2INCH = 0.0104166667
    return pixel * PIXEL2INCH

def get_color_array(cmap='rainbow', num_colors=8):
    if cmap == 'rainbow':
        cmap = cm.rainbow
    else:
        raise NotImplementedError
    COLOR_ARRAY = cmap(np.linspace(0, 1, num_colors))

    return COLOR_ARRAY


def visualize_bbox_xy(bbox,
                      fig=None, ax=None,
                      color=None, cmap='rainbow', num_colors=8, color_idx=None):
    if color is None:
        if color_idx is None:
            color_idx = 0
        color = get_color_array(cmap, num_colors)[color_idx]
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    patch = Rectangle(xy=(x1,y1),
                      width=(x2-x1), height=(y2-y1),
                      linewidth=1,
                      edgecolor=color,
                      facecolor='none')
    ax.add_patch(patch)
    ax.plot()
    return fig, ax


def visualize_bboxes_xy(bboxes_xy,
                        fig=None, ax=None,
                        color=None, cmap='rainbow', num_colors=8, color_idx=None,
                        ):
    '''
    bboxes_xy   : (num_bboxes, 4) where 4 containing (x1,y1,x2,y2)
    '''
    if color is None:
        if color_idx is None:
            color_idx = 0
        color = get_color_array(cmap, num_colors)[color_idx]

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    for bbox_xy in bboxes_xy:
        fig, ax = visualize_bbox_xy(bbox_xy, fig, ax, color, cmap, num_colors, color_idx)
    ax.plot()

    return fig, ax


def visualize_bbox_cxcy(bbox,
                        fig=None, ax=None,
                        color=None, cmap='rainbow', num_colors=8, color_idx=None):
    if color is None:
        if color_idx is None:
            color_idx = 0
        color = get_color_array(cmap, num_colors)[color_idx]
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    if torch.is_tensor(bbox):
        bbox = bbox.cpu().detach().numpy()
    cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    patch = Rectangle(xy=(cx - w / 2., cy - h / 2.),
                      width=w, height=h,
                      linewidth=1,
                      edgecolor=color,
                      facecolor='none')
    ax.add_patch(patch)
    ax.plot()
    return fig, ax


def visualize_bboxes_cxcy(bboxes,
                          fig=None, ax=None,
                          color=None, cmap='rainbow', num_colors=8, color_idx=None):
    if color is None:
        if color_idx is None:
            color_idx = 0
        color = get_color_array(cmap, num_colors)[color_idx]
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    for bbox_cxcy in bboxes:
        fig, ax = visualize_bbox_cxcy(bbox_cxcy, fig, ax, color, cmap, num_colors, color_idx)
    ax.plot()
    return fig, ax

