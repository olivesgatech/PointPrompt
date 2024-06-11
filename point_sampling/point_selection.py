import cv2
import numpy as np
import torch
from sklearn_extra.cluster import KMedoids
from typing import List
from PIL import Image
from scipy import ndimage
import sys

''' From https://github.com/SysCV/sam-pt/blob/874ff7e73d6ab05418a494d7a02ca233c0b31e8c/sam_pt/utils/query_points.py#L11 '''

def extract_random_mask_points(mask, n_points_to_select, mode):
    """
    Randomly select a specified number of points from the mask.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask tensor with shape (height, width) of dtype float32 with values in {0, 1}.
    n_points_to_select : int
        The number of points to select from the mask.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n_points_to_select, 2) containing the selected points. The dtype of the
        tensor is float32.
    """
    if mask.sum() == 0:
        print(f"Warning: mask.sum() == 0 in extract_random_mask_points")
        return torch.zeros((n_points_to_select, 2))
    if mode == 'inc':
        # Get nonzero pixel locations
        mask = torch.from_numpy(mask)
    else:
        # Get 'zero' locations
        mask = torch.from_numpy(1-mask)
    mask_pixels = mask.nonzero().float()
    assert len(mask_pixels) > 0
    if len(mask_pixels) < n_points_to_select:
        selected_points = mask_pixels.repeat(n_points_to_select // len(mask_pixels) + 1, 1)[:n_points_to_select]
    else:
        # Randomly select which pixels to query
        selected_points = mask_pixels[torch.randperm(len(mask_pixels))[:n_points_to_select]]

    selected_points = selected_points.flip(1)  # Change from (y, x) to (x, y)
    assert selected_points.shape == (n_points_to_select, 2)
    return selected_points
def extract_kmedoid_points(mask, n_points_to_select, subsample_size=1800, mode='inc'):
    """
    Randomly select the specified number of points from the mask using K-Medoids.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask tensor with shape (height, width) of dtype float32 with values in {0, 1}.
    n_points_to_select : int
        Number of points to select from the mask.
    subsample_size : int, optional
        Size of subsample to use for K-Medoids, by default 1800.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n_points_to_select, 2) containing the selected points. The dtype of the
        tensor is float32.
    """
    if mask.sum() == 0:
        print(f"Warning: mask.sum() == 0 in extract_kmedoid_points")
        return torch.zeros((n_points_to_select, 2))

    if mode == 'inc':
        # Get nonzero pixel locations
        mask = torch.from_numpy(mask)
    else:
        # Get 'zero' locations
        mask = torch.from_numpy(1-mask)

    mask_pixels = mask.nonzero().float()

    if len(mask_pixels) < n_points_to_select:
        selected_points = mask_pixels.repeat(n_points_to_select // len(mask_pixels) + 1, 1)[:n_points_to_select]
    else:
        # Sample N points from the largest cluster by performing K-Medoids with K=N
        mask_pixels = mask_pixels[torch.randperm(len(mask_pixels))[:subsample_size]]
        selected_points = KMedoids(n_clusters=n_points_to_select).fit(mask_pixels).cluster_centers_
        selected_points = torch.from_numpy(selected_points).type(torch.float32)

    # (y, x) -> (x, y)
    selected_points = selected_points.flip(1)

    assert selected_points.shape == (n_points_to_select, 2)
    return selected_points

def erode_mask_proportional_to_its_furthest_points_distance(mask: torch.Tensor,
                                                            erosion_percentage: float) -> torch.Tensor:
    """
    Erode the mask by a percentage of its diameter.

    Erode the mask by the specified percentage of its diameter.
    The diameter is computed as the distance between the two
    points that are the farthest from each other on the mask.
    The erosion is performed using a square kernel.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask tensor with shape (height, width) of dtype float32 with values in {0, 1}.
    erosion_percentage : float
        Percentage of the mask diameter to erode the mask by.

    Returns
    -------
    mask : torch.Tensor
        Eroded mask of shape (height, width).
    """
    #print('MAX OF MASK: ', mask.shape)
    mask_pixels = mask.nonzero().float()
    mask_diameter = torch.norm(mask_pixels.max(0)[0] - mask_pixels.min(0)[0]).item()
    erosion_size = int(mask_diameter * erosion_percentage)

    mask_for_cv = mask.cpu().numpy().astype(np.uint8)
    #print('masssk: ', mask_for_cv.shape)
    eroded_mask_for_cv = cv2.erode(mask_for_cv, np.ones((erosion_size, erosion_size), np.uint8), iterations=1)
    #print('ERODE: ', eroded_mask_for_cv)
    mask = torch.from_numpy(eroded_mask_for_cv).type(mask.dtype).to(mask.device)
    return mask

def extract_corner_points(image, mask, n_points_to_select, kmedoid_subsample_size=2000, mode='inc'):
    """
    Select a specified number of points from the mask using a corner detection algorithm. Erosion
    is applied on the mask at various levels if necessary, before performing corner detection,
    as to avoid selecting points on the edges of the mask.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor of shape (channels, height, width) and in uint8 [0-255] format.
    mask : torch.Tensor
        Binary mask tensor with shape (height, width) of dtype float32 with values in {0, 1}.
    n_points_to_select : int
        Number of points to select from the mask.
    kmedoid_subsample_size : int, optional
        Size of subsample to use for K-Medoids, by default 2000.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_points_to_select, 2) and dtype float32 containing the selected points.
        Points are in (x, y) format.
    """
    # put image in acceptable format [channels, height, width]

    image = image.permute(2, 0, 1)
    #print('----MASK: ', mask.shape)
    if mask.sum() == 0:
        print(f"Warning: mask.sum() == 0 in extract_corner_points")
        return torch.zeros((n_points_to_select, 2))

    # if mode == 'inc':
    #     # Get nonzero pixel locations
    #     mask = torch.from_numpy(mask.astype('uint8'))
    # else:
    #     # Get 'zero' locations
    #     mask = ~mask

    if mode != 'inc':
        mask = 1-mask
    #print('Img shape: ', image.shape)
    image = image.permute(1, 2, 0).cpu().numpy()
    mask_eroded = erode_mask_proportional_to_its_furthest_points_distance(torch.from_numpy(mask.astype('uint8')), erosion_percentage=0.06)
    if mask_eroded.sum() < 10:
        mask_eroded = erode_mask_proportional_to_its_furthest_points_distance(torch.from_numpy(mask.astype('uint8')), erosion_percentage=0.02)
    if mask_eroded.sum() < 10:
        mask_eroded = erode_mask_proportional_to_its_furthest_points_distance(torch.from_numpy(mask.astype('uint8')), erosion_percentage=0.01)
    if mask_eroded.sum() < 10:
        mask_eroded = torch.from_numpy(mask.astype('uint8'))

    mask_pixels = mask_eroded.nonzero().float()
    mask_diameter = torch.norm(mask_pixels.max(0)[0] - mask_pixels.min(0)[0]).item()
    #print('mask', mask_eroded.shape)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #print('gray img shape: ', gray_image.shape)
    corner_points = cv2.goodFeaturesToTrack(
        image=gray_image,
        maxCorners=n_points_to_select,
        qualityLevel=0.001,
        minDistance=mask_diameter / n_points_to_select,
        mask=mask_eroded.cpu().numpy().astype(np.uint8),
        blockSize=3,
        gradientSize=3,
    )
    if corner_points is None:
        corner_points = np.empty((0, 1, 2))
    corner_points = torch.from_numpy(corner_points).type(torch.float32).squeeze(1)

    if len(corner_points) < n_points_to_select:
        # Replace the missing points with K-medoid points
        n_missing_points = n_points_to_select - corner_points.shape[0]
        kmedoid_points = extract_kmedoid_points(mask, n_missing_points, subsample_size=kmedoid_subsample_size)

        corner_points = torch.cat((corner_points, kmedoid_points), dim=0)
    #print('s', corner_points.shape)
    assert corner_points.shape == (n_points_to_select, 2)
    return corner_points

# From https://github.com/yhydhx/SAMAug/blob/a7dce8878d56d2a265bd2819de3246c726e4adb4/SAMAug.py#L92
"""Max Entropy Point"""
def image_entropy(image):
    # Convert the image to grayscale
    check = image.flatten().max()
    max_val_r = 256
    max_val_bins = max_val_r
    # if check > 1:
    #     max_val_r = 256
    #     max_val_bins = max_val_r
    # else:
    #     print('Binary')
    #     max_val_r = 1
    #     max_val_bins = 2
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [max_val_bins], [0, max_val_r])
    # Normalize the histogram
    hist /= hist.sum()
    # Calculate the entropy
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy

def calculate_image_entroph(img1, img2):
    # Calculate the entropy for each image
    entropy1 = image_entropy(img1)
    try:
        entropy2 = image_entropy(img2)
    except:
        entropy2 = 0
    # Compute the entropy between the two images
    entropy_diff = abs(entropy1 - entropy2)
    # print("Entropy Difference:", entropy_diff)
    return entropy_diff

def select_grid(image, center_point, grid_size):
    (img_h, img_w, _) = image.shape

    # Extract the coordinates of the center point
    x, y = center_point
    #print('x: ', x)
    x = int(np.floor(x))
    y = int(np.floor(y))
    # Calculate the top-left corner coordinates of the grid
    top_left_x = x - (grid_size // 2) if x - (grid_size // 2) > 0 else 0
    top_left_y = y - (grid_size // 2) if y - (grid_size // 2) > 0 else 0
    bottom_right_x = top_left_x + grid_size if top_left_x + grid_size < img_w else img_w
    bottom_right_y = top_left_y + grid_size if top_left_y + grid_size < img_h else img_h

    # Extract the grid from the image
    grid = image[top_left_y: bottom_right_y, top_left_x: bottom_right_x]

    return grid

def get_entropy_points(input_point, mask, image, n_points):

    entropy_all = []
    grid_pts_all = []
    selected_points = []

    max_entropy_point = [0,0]
    max_entropy = 0
    grid_size = 9
    center_grid = select_grid(image, input_point, grid_size)

    # Sample a portion of total pixels
    inds = np.argwhere(mask == True)
    amt = int(0.15*len(inds))
    base = 0.5
    while (amt < n_points) and (base <= 1.0):
        print('too small')
        amt = int(base*len(inds))
        base += 0.1
    if (amt < n_points):
        print('Still too low')
        amt = len(inds)
        print('Amt: ', amt)
        n_points = len(inds)
    i = np.random.choice(np.random.permutation(np.arange(len(inds))), size=amt, replace=False)
    indices = inds[i]

    for x,y in indices:
        grid = select_grid(image, [x,y], grid_size)
        entropy_diff = calculate_image_entroph(center_grid, grid)
        grid_pts_all.append([x,y])
        entropy_all.append(entropy_diff)
        if entropy_diff > max_entropy:
            max_entropy_point = [x,y]
            #print('new max entropy pt: ', max_entropy_point)
            max_entropy = entropy_diff

    if n_points != 1:
        for pt in range(n_points):
            idx = entropy_all.index(max(entropy_all))
            grid_pt = grid_pts_all[idx]
            selected_points.append([grid_pt[1], grid_pt[0]])
            # remove max from list and get 2nd, 3rd, 4th, etc max
            _ = entropy_all.pop(idx)
            _ = grid_pts_all.pop(idx)
        # print('----')
        # print(selected_points)

    else:
        selected_points.append([max_entropy_point[1], max_entropy_point[0]])
    #print('Selected pts: ', selected_points)
    return torch.Tensor(selected_points)

def get_distance_points(input_point, mask, n_points):
    dist_all = []
    grid_pts_all = []
    selected_points = []

    max_distance_point = [0,0]
    max_distance = 0

    # Sample a portion of total pixels
    inds = np.argwhere(mask == True)
    amt = int(0.15 * len(inds))
    base = 0.5

    while (amt < n_points) and (base <= 1.0):
        print('too small')
        amt = int(base*len(inds))
        base += 0.1
    if (amt < n_points):
        print('Still too low')
        amt = len(inds)
        print('Amt: ', amt)
        n_points = len(inds)
    i = np.random.choice(np.arange(len(inds)), size=amt, replace=False)
    indices = inds[i]

    for x,y in indices:
        distance = np.sqrt((x- input_point[0])**2 + (y- input_point[1]) ** 2)
        grid_pts_all.append([x, y])
        dist_all.append(distance)
        if max_distance < distance:
            max_distance_point = [x,y]
            max_distance = distance

    if n_points != 1:
        for pt in range(n_points):
            idx = dist_all.index(max(dist_all))
            grid_pt = grid_pts_all[idx]
            selected_points.append([grid_pt[1], grid_pt[0]])
            # remove max from list and get 2nd, 3rd, 4th, etc max
            _ = dist_all.pop(idx)
            _ = grid_pts_all.pop(idx)

    else:
        selected_points.append([max_distance_point[1],max_distance_point[0]])

    return torch.Tensor(selected_points)

def object_sample(mask, n_points, subsample_size=1800):
    # Given the mask, find the number of objects per image
    distinct, num_objects = ndimage.label(mask.astype(bool))
    # make sure bad labeling doesn't affect number of objects
    red = 0
    for j in range(num_objects):
        pts = np.where(distinct == j+1)
        if len(pts[0]) <= 3:
            red += 1
            print('reduced by ', red)
            distinct[pts] = 0
    num_objects -= red
    inds = np.where(distinct != 0)
    distinct[inds] = 1
    distinct, num_objects = ndimage.label(distinct.astype(bool))

    #print('Number of objects: ', num_objects)
    num_pts_pr_obj = int(n_points/num_objects)

    # Determine if there are any leftover points
    _, count = np.unique(distinct.flatten(), return_counts=True)
    keys = list(_)[1:]
    count = count[1:]
    point_count = [num_pts_pr_obj]*num_objects
    # assign higher number of points to portions of the mask that are larger
    remain = n_points - (num_pts_pr_obj * num_objects)

    if num_pts_pr_obj < 1:
        # more objects than points
        point_count = [0]*num_objects
        for pt in range(n_points):
            large = np.argmax(count)
            count[large] = 0
            target = point_count[large] + 1
            point_count[large] = target

    else:
        if remain > 0:
            # Determine how many points to assign each object in mask
            for pt in range(remain):
                large = np.argmax(count)
                count[large] = 0
                target = point_count[large] + 1
                point_count[large] = target

    amount = dict(zip(keys, point_count))
    # Get points
    for i in range(num_objects):
        current_mask = np.copy(distinct)
        # Get only a certain portion of mask depending on object number
        current_mask[np.where(current_mask != i+1)] = 0
        current_mask = torch.from_numpy(current_mask)
        mask_pixels = current_mask.nonzero().float()
        mask_pixels = mask_pixels[torch.randperm(len(mask_pixels))[:subsample_size]]
        # Perform K-Mediods to select points within specific region
        if amount[i+1] == 0:
            continue
        else:
            if len(mask_pixels) < n_points:
                points_temp = mask_pixels.repeat(n_points // len(mask_pixels) + 1, 1)[:n_points]
            else:
                points_temp = KMedoids(n_clusters=amount[i+1]).fit(mask_pixels).cluster_centers_
            try:
                selected_points = np.concatenate((selected_points, points_temp))
            except NameError:
                selected_points = np.copy(points_temp)

    selected_points = torch.from_numpy(selected_points).type(torch.float32)
    selected_points = selected_points.flip(1)
    return selected_points

def perimeter_selection(mask, n_points):
    # Increase ground truth mask size by performing dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate = cv2.dilate(mask, kernel, iterations=1)
    # Subtract original mask and dilated mask to get area where points can be selected
    new_mask = dilate - mask
    # New mask represents viable pixels close to border of ground truth mask
    new_mask = torch.from_numpy(new_mask)
    mask_pixels = new_mask.nonzero().float()

    assert len(mask_pixels) > 0
    if len(mask_pixels) < n_points:
        selected_points = mask_pixels.repeat(n_points // len(mask_pixels) + 1, 1)[:n_points]
    else:
        # Randomly select which pixels to query
        selected_points = mask_pixels[torch.randperm(len(mask_pixels))[:n_points]]

    #selected_points = torch.from_numpy(selected_points).type(torch.float32)
    selected_points = selected_points.flip(1)
    return selected_points.numpy()

def max_dist_modified(mask, n_points):
    height, width = mask.shape
    pos_h, pos_w = np.where(mask > 0)
    center_h = int(np.average(pos_h))
    center_w = int(np.average(pos_w))
    dict_ = {}
    side_1 = np.array([0, int(width/2)]) # top
    line_1 = {'h': np.array([0]), 'w': np.arange(0, width)}
    dict_[0] = line_1
    side_2 = np.array([int(height/2), width]) # right
    line_2 = {'h': np.arange(height), 'w': np.array([width-1])}
    dict_[1] = line_2
    side_3 = np.array([height, int(width/2)]) # bottom
    line_3 = {'h': np.array([height-1]), 'w': np.arange(0, width)}
    dict_[2] = line_3
    side_4 = np.array([int(height/2), 0]) # left
    line_4 = {'h': np.arange(height), 'w': np.array([0])}
    dict_[3] = line_4
    sides = np.vstack((side_1, side_2))
    sides = np.vstack((sides, side_3))
    sides = np.vstack((sides, side_4))

    max_dist = 0
    for j in range(len(sides)):
        input_point = sides[j]
        distance = np.sqrt((center_h - input_point[0]) ** 2 + (center_w - input_point[1]) ** 2)
        if distance > max_dist:
            max_dist = distance
            max_point = input_point
            print('MAX: ', max_point)
            best_line = dict_[j]

    #
    # h_min = max_point[0] - 50 if (max_point[0]-50) > 0 else max_point[0]
    # h_max = max_point[0] + 50 if (max_point[0]+50) < height else max_point[0]
    # w_min = max_point[1] - 50 if (max_point[1]-50) > 0 else max_point[1]
    # w_max = max_point[1] + 50 if (max_point[1] + 50) < width else max_point[1]

    inds = np.where(mask > 0)
    mask[inds] = 0

    if len(best_line['w']) == 1:
        min_w = best_line['w'][0] - 20 if (best_line['w'][0] - 20) > 0 else best_line['w'][0]
        max_w = best_line['w'][0] + 20 if (best_line['w'][0] + 20) < width else best_line['w'][0]
        mask[best_line['h'], min_w:max_w] = 1
    elif len(best_line['h']) == 1:
        min_ = best_line['h'][0] - 20 if (best_line['h'][0] - 20) > 0 else best_line['h'][0]
        max_ = best_line['h'][0] + 20 if (best_line['h'][0] + 20) < height else best_line['h'][0]
        mask[min_:max_, best_line['w']] = 1


    mask = torch.from_numpy(mask)
    mask_pixels = mask.nonzero().float()

    selected_points = mask_pixels[torch.randperm(len(mask_pixels))[:n_points]]
    selected_points = selected_points.flip(1)
    return selected_points.numpy()

