"""
This file contains utilities for plotting Segment-Anything masks and Prompt Points. 
This code is adapted from the segment anything code (https://github.com/facebookresearch/segment-anything)
"""
import numpy as np

def show_mask(mask, ax, random_color=False, alpha=0.6):
    """
    This adds a mask on top of the existing matplotlib axis
    Parameters:
        - mask: Mask to plot
        - ax: matplotlib axis object to add plot to
        - random_color (bool): If True, choose a random color for the mask. If False, plot a red mask. 
        - alpha (float): Alpha transparency of mask. 1.0 is fully opaque and 0.0 is fully transparent (won't show anything)
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 0/255, 0/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    """
    Adds the prompting points as red and green stars on an existing matplotlib axis. 
    Parameters:
        - coords: Coordinates of the points
        - labels: Whether the points are positive or negative points. 1 is a positive point and 0 is a negative point. 
        - ax: matplotlib axis object to plot on
        - marker_size: size of each marker
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
