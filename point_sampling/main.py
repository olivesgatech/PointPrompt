import os
from segment_anything import sam_model_registry, SamPredictor
import argparse
import glob

from intelligent_point_selection import get_saliency_point
from point_selection import *

# Number of best inclusion / exclusion points recorded from annotators [inclusion, exclusion]
point_nums = {
    'Baseball bat': [4, 2],
    'Bird': [9, 1],
    'Bus': [7, 1],
    'Cat': [6, 1],
    'Clock': [8, 3],
    'Cow': [16, 4],
    'Dog': [8, 2],
    'Tie': [4, 2],
    'Stop sign': [8, 1],
    'Dolphin above': [7, 2],
    'Dolphin below': [10, 4],
    'Polyp': [14, 2],
    'Skin': [11, 2],
    'Breast': [4, 1],
    'Salt dome': [5, 3],
    'Chalk group': [8, 5]
}


def parse():
    parser = argparse.ArgumentParser(description="point sampling")
    parser.add_argument('--query_strategy', type=str, default='saliency',
                        choices=['rand', 'shi-tomasi', 'kmediod', 'entropy', 'max_dist', 'saliency', 'obj'])
    parser.add_argument('--img_dir', type=str, default='dataset_path')
    parser.add_argument('--home_dir', type=str, default='path_to_home') # For pretrained stuff
    parser.add_argument('--results_dir', type=str, default='/example_results_path/')
    parser.add_argument('--num_pts_mode', type=str, default='avg')
    args = parser.parse_args()
    return args

def main():

    np.random.seed(2024)
    torch.manual_seed(2024)

    args = parse()
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=args.home_dir + sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    img_dir = args.img_dir
    results_directory = args.results_dir
    query = args.query_strategy
    categories = glob.glob(img_dir + '*') # Can change this to specific folders
    # Iterate through all data folders
    for cat in categories:

        cur = cat.split('/')[-1]
        print('Category: ', cur)
        # Make sure all directories exist; create them if not
        results_folder = results_directory + cur + '/' + query + '/'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if args.num_pts_mode == 'avg':
            ex = ''
        else:
            ex = '_1pt'
        quant_folder = results_folder + 'Quantitative' + ex + '/'
        mask_folder = results_folder + 'Mask' + ex + '/'
        points_folder = results_folder + 'Points' + ex + '/'
        if not os.path.exists(quant_folder):
            os.makedirs(quant_folder)
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)
        if not os.path.exists(points_folder):
            os.makedirs(points_folder)
        # Load Ground Truth Labels
        ground_truth = np.load(cat + '/labels.npy', allow_pickle=True)
        save_mask = []
        inclusion_pt = []
        exclusion_pt = []
        tot_scores = []
        imgs = np.load(cat + '/samples.npy', allow_pickle=True)

        # for each image, select sample points
        for i in range(len(ground_truth)): #len(ground_truth)
            #print('img: ', i)
            if args.num_pts_mode == 'avg':
                # 'Avg' refers to using the average inclusion/exclusion points used for annotators
                inclusion = point_nums[cur][0]
                exclusion = point_nums[cur][1]
            else:
                # Sample only one new inclusion/exclusion point
                inclusion = 1
                exclusion = 1


            current = ground_truth[i]
            # check format of ground truth mask
            if max(current.flatten()) > 1:
                current = np.array(current / current.max(), dtype=np.uint8)


            img = imgs[i]

            if cur == 'Breast':
                if i == 321:
                    current = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
            elif cur == 'Polyp':
                current = current.astype(np.uint8)
            elif cur == 'Dolphin below':
                current = current.astype(np.uint8)
                img *= 255
                img = img.astype(np.uint8)
            elif cur == 'Dolphin above':
                current = current.astype(np.uint8)
                img *= 255
                img = img.astype(np.uint8)
            elif cur == 'Chalk group':
                current = current.astype(np.uint8)
                img *= 255
                img = img.astype(np.uint8)
            elif cur == 'Salt dome':
                current = current.astype(np.uint8)
                img *= 255
                img = img.astype(np.uint8)

            if len(img.shape) == 2:
                img = cv2.cvtColor((np.array(((img + 1) / 2) * 255, dtype='uint8')), cv2.COLOR_GRAY2RGB)

            img_ = torch.from_numpy(img)
            predictor.set_image(img)
            # query points
            if query == 'rand':
                inc_points = extract_random_mask_points(mask=current, n_points_to_select=inclusion, mode='inc')
                ex_points = extract_random_mask_points(mask=current, n_points_to_select=exclusion, mode='ex')

            elif query == 'shi-tomasi':
                inc_points = extract_corner_points(image=img_, mask=current, n_points_to_select=inclusion, mode='inc')
                ex_points = extract_corner_points(image=img_, mask=current, n_points_to_select=exclusion,
                                                   mode='ex')

            elif query == 'entropy':
                inc_init = extract_random_mask_points(mask=current, n_points_to_select=1, mode='inc')
                ex_init = extract_random_mask_points(mask=current, n_points_to_select=1, mode='ex')

                inc_points_ = get_entropy_points(input_point=inc_init.numpy()[0], mask=current, image=img,
                                                n_points=inclusion)
                ex_points_ = get_entropy_points(input_point=ex_init.numpy()[0], mask=1-current, image=img,
                                               n_points=exclusion)
                inc_points = np.concatenate((inc_init, inc_points_))
                ex_points = np.concatenate((ex_init, ex_points_))

            elif query == 'max_dist':
                inc_init = extract_random_mask_points(mask=current, n_points_to_select=1, mode='inc')
                ex_init = extract_random_mask_points(mask=current, n_points_to_select=1, mode='ex')

                inc_points_ = get_distance_points(input_point=inc_init.numpy()[0], mask=current, n_points=inclusion)
                ex_points_ = get_distance_points(input_point=ex_init.numpy()[0], mask=1-current, n_points=exclusion)

                inc_points = np.concatenate((inc_init, inc_points_))
                ex_points = np.concatenate((ex_init, ex_points_))

            elif query == 'saliency':
                # We first randomly query a point from GT mask and get associated SAM output
                draw_mask = np.array([])
                while len(draw_mask) == 0:
                    inc_init = extract_random_mask_points(mask=current, n_points_to_select=1, mode='inc') # --> (x,y)
                    ex_init = extract_random_mask_points(mask=current, n_points_to_select=1, mode='ex')
                    input_ = np.concatenate(([1], [0]))
                    pts = np.concatenate((inc_init.numpy(), ex_init.numpy()))
                    masks, _, _ = predictor.predict(
                        point_coords=pts,
                        point_labels=input_,
                        multimask_output=True,
                    )
                    # Computed mask given initial point. Ensure that mask isn't empty
                    Original_Mask = masks[0]
                    draw_mask = np.argwhere(Original_Mask > 0)

                inc_points_, vst_inc = get_saliency_point(img=img, mask=Original_Mask, n_points=inclusion, mode='inc')
                ex_points_, vst_ex = get_saliency_point(img=img, mask=1-Original_Mask, n_points=exclusion, mode='ex')
                ex_points = np.concatenate((ex_init, ex_points_))
                # concat
                inc_points = np.concatenate((inc_init, inc_points_))
            else:
                inc_points = extract_kmedoid_points(mask=current, n_points_to_select=inclusion, mode='inc')
                ex_points = extract_kmedoid_points(mask=current, n_points_to_select=exclusion, mode='ex')

            points = np.concatenate((inc_points, ex_points))
            input_label = np.concatenate(([1] * len(inc_points), [0] * len(ex_points)))

            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=input_label,
                multimask_output=True,
            )
            # Computed mask
            mask = masks[0].astype('int')
            intersection = (mask & current).sum()
            union = (mask | current).sum()
            if intersection == 0:
                score = 0
            else:
                score = intersection / union

            tot_scores.append(score)
            # save mask
            save_mask.append(mask)
            # Save points
            if torch.is_tensor(inc_points):
                inclusion_pt.append(inc_points.numpy())
            else:
                inclusion_pt.append(inc_points)
            if torch.is_tensor(ex_points):
                exclusion_pt.append(ex_points.numpy())
            else:
                exclusion_pt.append(ex_points)

        print('----')
        print('Mean IOU: ', np.mean(np.array(tot_scores)))
        # Save IOU
        np.save(quant_folder + 'iou.npy', np.array(tot_scores))
        # Save stuff
        np.save(points_folder + 'inclusion_points.npy', np.array(inclusion_pt, dtype=object), allow_pickle=True)
        np.save(points_folder + 'exclusion_points.npy', np.array(exclusion_pt, dtype=object), allow_pickle=True)
        # Save resulting mask
        np.save(mask_folder + 'masks.npy', np.array(save_mask, dtype=object), allow_pickle=True)



# start main
if __name__ == "__main__":
    main()
