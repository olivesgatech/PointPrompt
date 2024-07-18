import numpy as np
import os
import cv2
import pandas as pd

from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.feature import graycomatrix, graycoprops

from utils import points_to_mask_dist, points_are_collinear, crop_to_object, get_mask_diameter, chamfer_distance, max_point_dist, avg_min_point_dist


def get_gclm_props(image, mask, mode='obj'):

    if np.sum(mask) < mask.shape[0] * mask.shape[1] * 0.0005:
        return None, None, None, None, None, None

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mode=='obj':
        diameter = get_mask_diameter(mask)
        d = int(diameter*0.01) if int(diameter*0.01) > 0 else 1
    else:
        diag = np.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2)
        d = int(diag*0.01) if diag>=100 else 1
    masked = (image*mask).astype('uint16')
    masked[mask == 0] = 256
    g = graycomatrix(crop_to_object(masked), [d], [0, np.pi / 2], levels=257)
    g = g[:256, :256, :, :]
    contrast = np.sum(graycoprops(g, 'contrast'))
    dissimilarity = np.sum(graycoprops(g, 'dissimilarity'))
    homogeneity = np.sum(graycoprops(g, 'homogeneity'))
    energy = np.sum(graycoprops(g, 'energy'))
    correlation = np.sum(graycoprops(g, 'correlation'))
    asm = np.sum(graycoprops(g, 'ASM'))
    return contrast, dissimilarity, homogeneity, energy, correlation, asm


def get_general_stats(path, im_index, timestamp):
    g = np.load(path + '/points/' + str(im_index) + '_green.npy', allow_pickle=True)
    r = np.load(path + '/points/' + str(im_index) + '_red.npy', allow_pickle=True)
    score = np.load(path + '/scores/' + str(im_index) + 'score.npy', allow_pickle=True)

    return len(g[timestamp]) + len(r[timestamp]), len(g[timestamp]), len(r[timestamp]), score[timestamp]


def get_data_stats(image, mask):
    size = np.sum(mask) / (mask.shape[0] * mask.shape[1])

    labeled, cc_density = ndimage.label(mask.astype(bool))
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    masked_image = gray_image * mask
    edges = cv2.Canny(masked_image, 100, 200)
    binary_array = (edges > 0).astype(np.uint8)
    obj_char = np.sum(binary_array*mask) / np.sum(mask)

    return size, cc_density, obj_char


def get_prompt_stats(mask, path, im_index, timestamp):
    if np.sum(mask) < mask.shape[0] * mask.shape[1] * 0.0005:
        return None, None, None, None, None, None, None, None

    green_prompts = np.load(path + '/points/' + str(im_index) + '_green.npy', allow_pickle=True)[timestamp]
    red_prompts = np.load(path + '/points/' + str(im_index) + '_red.npy', allow_pickle=True)[timestamp]

    if len(green_prompts) < 3:
        prompt_coverage = 0
    else:
        if points_are_collinear(green_prompts):
            prompt_coverage = 0
        else:
            binary_image = np.zeros(mask.shape, dtype=np.uint8)
            hull = ConvexHull(green_prompts)
            hull_points = green_prompts[hull.vertices]
            cv2.fillPoly(binary_image, [hull_points.astype('int64')], 1)
            prompt_coverage = np.sum(binary_image & mask) / np.sum(mask)

    max_dist = max_point_dist(green_prompts)
    diameter = get_mask_diameter(mask)
    max_spread = max_dist / diameter

    avg_min_dist = avg_min_point_dist(green_prompts)
    n_min_dist = avg_min_dist / diameter

    d = chamfer_distance(green_prompts, red_prompts)
    inc_exc_dist = d / (np.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2))

    max_dist = max_point_dist(red_prompts)
    diagonal = np.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2)
    max_spread_exc = max_dist / diagonal

    inc_efficiency = prompt_coverage/len(green_prompts)
    exc_encasing = points_to_mask_dist(red_prompts, mask, mode='exc')/diagonal
    inc_encasing = points_to_mask_dist(green_prompts, mask, mode='inc')/diameter

    return prompt_coverage, max_spread, n_min_dist, inc_exc_dist, max_spread_exc, inc_efficiency, exc_encasing, inc_encasing

def get_ind_prompt_dict(mask, path, im_index, timestamp):
    ind_dict = {}
    ind_dict['Prompt coverage'], ind_dict['Max spread'], ind_dict['Avg min distance'], ind_dict['Inc/Exc distance'], ind_dict[
        'Max spread exc'], ind_dict['Inc efficiency'], i_dict['Exc encasing'], i_dict['Inc encasing'] = get_prompt_stats(mask, path, im_index, timestamp)
    return ind_dict


datasets = ['Dolphin above', 'Dolphin below', 'Salt dome', 'Chalk group', 'Baseball bat', 'Bird', 'Bus', 'Cat', 'Clock',
            'Cow', 'Dog', 'Tie', 'Stop sign',  'Polyp', 'Skin', 'Breast']


def parse():
    parser = argparse.ArgumentParser(description="feature extraction")
    parser.add_argument('--data_path', type=str, default='Image datasets')
    parser.add_argument('--prompts_path', type=str, default='Prompting results')


    args = parser.parse_args()
    return args

def main():
    args = parse()
    data = []

    object_counts_df = pd.read_csv('object_counts.csv')

    for ds in datasets:
        path = args.prompts_path + '/' + ds

        student_folders = [f.path for f in os.scandir(path) if f.is_dir()]

        ds_path = os.path.join(args.data_path, ds)
        masks = np.load(os.path.join(ds_path, 'labels.npy'), allow_pickle=True)
        images = np.load(os.path.join(ds_path, 'samples.npy'), allow_pickle=True)
        data_stats = []
        print('Processing ' + ds)
        for idx in range(len(masks)):
            size, cc_density, obj_char = get_data_stats(images[idx], masks[idx])
            contrast, dissimilarity, homogeneity, energy, correlation, asm = get_gclm_props(images[idx], masks[idx])
            b_contrast, b_dissimilarity, b_homogeneity, b_energy, b_correlation, b_asm = get_gclm_props(images[idx], 1-masks[idx], mode='bckg')
            data_stats.append((size, cc_density, obj_char, contrast, dissimilarity, homogeneity, energy, correlation, asm, b_contrast, b_dissimilarity, b_homogeneity, b_energy, b_correlation, b_asm))

        if object_counts_df['Dataset'].str.contains(ds, na=False).any():
            obj_count_ds = object_counts_df[object_counts_df['Dataset'] == ds]
        else:
            obj_count_ds = None

        for st_path in student_folders:
            print(st_path)
            templist = [f for f in os.listdir(st_path + '/eachround') if
                        os.path.isfile(os.path.join(st_path + '/eachround', f))]
            num_files = len(templist)
            for i in range(num_files):
                i_dict = {'Dataset': ds, 'Student': st_path, 'Image ID': i}

                fn = st_path + '/eachround/' + str(i) + '_.npy'
                rounds = np.load(fn)
                i_dict['Number of rounds'] = len(np.unique(rounds))

                fn = st_path + '/sorts/' + str(i) + '_sort.npy'
                sort = np.load(fn)
                if len(sort) > 0:
                    # General-level stats
                    i_dict['Best round'] = rounds[sort[0]]

                    i_dict['Best round length'] = np.sum(rounds == rounds[sort[0]])

                    i_dict['Best timestamp'] = sort[0] if rounds[sort[0]] == 0 else sort[0] - np.sum(rounds == 0)

                    first_timestamp = 0 if rounds[sort[0]] == 0 else np.sum(rounds == 0)
                    (i_dict['First # points'], i_dict['First # green points'], i_dict['First # red points'],
                     i_dict['First score']) = get_general_stats(st_path, i, first_timestamp)

                    (i_dict['Best # points'], i_dict['Best # green points'], i_dict['Best # red points'],
                     i_dict['Best score']) = get_general_stats(st_path, i, sort[0])

                    last_timestamp = len(rounds) - 1 if rounds[sort[0]] == rounds[-1] else np.sum(rounds == 0) - 1
                    (i_dict['Last # points'], i_dict['Last # green points'], i_dict['Last # red points'],
                     i_dict['Last score']) = get_general_stats(st_path, i, last_timestamp)

                    i_dict['Best-first gap'] = i_dict['Best score'] - i_dict['First score']

                    # Data-level stats
                    (i_dict['Mask size'], i_dict['CC Density'], i_dict['Texture score'], i_dict['Contrast'],
                     i_dict['Dissimilarity'], i_dict['Homogeneity'], i_dict['Energy'], i_dict['Correlation'],
                     i_dict['ASM'], i_dict['BK Contrast'], i_dict['BK Dissimilarity'], i_dict['BK Homogeneity'],
                     i_dict['BK Energy'], i_dict['BK Correlation'], i_dict['BK ASM']) = data_stats[i]
                    if obj_count_ds is not None:
                        i_dict['Obj Density'] = obj_count_ds['Objects Count'].iloc[i]
                    else:
                        i_dict['Obj Density'] = 1

                    i_dict['Merged'] = int(i_dict['Obj Density'] > i_dict['CC Density'])
                    i_dict['Split'] = int(i_dict['Obj Density'] < i_dict['CC Density'])
                    i_dict['Compact'] = int(i_dict['Obj Density'] == i_dict['CC Density'])

                    for j in range(len(sort)):
                        # Prompt-level stats

                        pref_dict = i_dict.copy()
                        pref_dict['Current timestamp'] = j
                        pref_dict['Current round'] = rounds[j]
                        (pref_dict['Current # points'], pref_dict['Current # green points'], pref_dict['Current # red points'],
                         pref_dict['Current score']) = get_general_stats(st_path, i, j)

                        pref_dict['Current is best'] = int(j == sort[0])
                        ind_dict = get_ind_prompt_dict(masks[i], st_path, i, j)
                        pref_dict.update(ind_dict)
                        data.append(pref_dict)


    df = pd.DataFrame(data)
    df.to_csv('stats.csv', index=False)
