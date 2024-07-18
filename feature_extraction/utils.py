import numpy as np
import os
import cv2
import pandas as pd

from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.feature import graycomatrix, graycoprops
from scipy.signal import find_peaks




def points_to_mask_dist(points, mask, mode='exc'):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_dists = []
    factor = -1 if mode == 'exc' else 1
    for point in points:
        dists = []
        for i, contour in enumerate(contours):
            dist = factor*cv2.pointPolygonTest(contour, point.astype('float'), True)
            dists.append(dist)
        min_dists.append(np.min(np.array(dists)))

    avg_dist = np.mean(min_dists)
    out = avg_dist if avg_dist > 0 else 0
    return out


def crop_to_object(image):
    # Find the coordinates of non-zero pixels
    non_zero_coords = np.argwhere(image != 0)

    # Find the bounding box of the non-zero pixels
    top_left = non_zero_coords.min(axis=0)
    bottom_right = non_zero_coords.max(axis=0)

    # Crop the image using the bounding box coordinates
    cropped_image = image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

    return cropped_image
    
def get_mask_diameter(mask):
    points = np.column_stack(np.where(mask == 1))
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    diameter = max_point_dist(hull_points)
    return diameter


def points_are_collinear(points):
    # Extract the first point
    x1, y1 = points[0]

    # Iterate over pairs of points (starting from the second point)
    for i in range(1, len(points) - 1):
        x2, y2 = points[i]
        x3, y3 = points[i + 1]

        # Calculate the cross product of vectors (x2 - x1, y2 - y1) and (x3 - x1, y3 - y1)
        cross_product = (y2 - y1) * (x3 - x1) - (y3 - y1) * (x2 - x1)

        # If cross product is not zero, points are not collinear
        if cross_product != 0:
            return False

    # If all cross products are zero, points are collinear
    return True


def chamfer_distance(P1, P2):
    dists1 = []
    for i in range(len(P1)):
        d_min = np.min(np.sqrt(np.sum((P1[i] - P2) ** 2, axis=1).astype('int64')))
        dists1.append(d_min)

    dists2 = []
    for i in range(len(P2)):
        d_min = np.min(np.sqrt(np.sum((P2[i] - P1) ** 2, axis=1).astype('int64')))
        dists2.append(d_min)

    return (np.mean(np.array(dists1)) + np.mean(np.array(dists2))) / 2


def max_point_dist(Ps):
    dists = []
    for i in range(len(Ps)):
        d_max = np.max(np.sum((Ps[i] - Ps) ** 2, axis=1))
        dists.append(d_max)

    return np.sqrt(np.max(np.array(dists)))


def avg_min_point_dist(Ps):
    dists = []
    for i in range(len(Ps)):
        d_sort = np.sort(np.sum((Ps[i] - Ps) ** 2, axis=1))
        if len(d_sort)>1:
            dists.append(d_sort[1])
        else:
            dists.append(0)

    return np.sqrt(np.mean(np.array(dists)))


def dom_colors(mask, image, n_colors=2):
    #Image (masked) histogram
    masked = mask*image
    hist, bins = np.histogram(masked.ravel(), 256, [0,256])
    hist[0] = 0
    hist = hist/np.sum(mask)
    hist = np.convolve(hist, np.ones(10) / 10, mode='valid')

    #Finding top peak values
    peaks, props = find_peaks(hist)
    peak_values = hist[peaks]
    top_peaks_indices = peaks[np.argsort(peak_values)[::-1][:n_colors]]

    #locating corresponding colors
    values = []
    for intensity in top_peaks_indices:
        coordinates = np.argwhere(masked == intensity)
        idx = np.random.choice(len(coordinates))
        coord = coordinates[idx]
        value = image[coord[0], coord[1]]
        values.append(value)

    return values