import argparse

import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
from torch.backends import cudnn
import torch.nn.functional as F
from models.ImageDepthNet import ImageDepthNet
from PIL import Image
import numpy as np

def get_saliency_point(img, mask, n_points, mode='inc'):

    (img_h, img_w, _) = img.shape
    coor = np.argwhere(mask > 0)
    if mode == 'inc':
        ymin = min(coor[:, 0])
        ymax = max(coor[:, 0])
        xmin = min(coor[:, 1])
        xmax = max(coor[:, 1])

        xmin2 = xmin - 10 if xmin - 10 > 0 else 0
        xmax2 = img_w if xmax + 10 > img_w else xmax + 10
        ymin2 = ymin - 10 if ymin - 10 > 0 else 0
        ymax2 = img_h if ymax + 10 > img_h else ymax + 10

        vst_input_img = img[ymin2:ymax2, xmin2:xmax2, :] #--> input is height, width (y,x)

    else:
        ymin = min(coor[:, 0])
        ymax = max(coor[:, 0])
        xmin = min(coor[:, 1])
        xmax = max(coor[:, 1])
        idxs_mask = np.array([])
        coverage = 100
        # Ensure at least some background is selected by ensuring it is > 10% of the total selection
        while len(idxs_mask) < coverage:
            xmin2 = np.random.choice(np.arange(xmin, xmax-10))
            xmax2 = np.random.choice(np.arange(xmin2+10, xmax))
            if xmax2 - xmin2 < 100:
                xmax2 = xmax2 + 100 if xmax2+100 < img_w else xmax2
            ymin2 = np.random.choice(np.arange(ymin, ymax-10))
            ymax2 = np.random.choice(np.arange(ymin2+10, ymax))
            if ymax2 - ymin2 < 100:
                ymax2 = ymax2 + 100 if ymax2+100 < img_h else ymax2
            # Ensure at least some of background is selected
            size = (xmax2-xmin2)*(ymax2-ymin2)
            mask_check = mask[ymin2:ymax2, xmin2:xmax2]
            idxs_mask = np.argwhere(mask_check > 0)
            coverage = 0.1*size

        vst_input_img = img[ymin2:ymax2, xmin2:xmax2, :]


    # VST mask
    vst_mask = VST(img_npy=vst_input_img) # --> output is (width, height)
    # judge point in the vst mask
    vst_indices = np.argwhere(vst_mask > 0)
    # select points
    vst_roi_random_point = []
    vst_random_point = []
    if n_points != 1:
        if len(vst_indices) < n_points:
            n_points = len(vst_indices)
        random_index = np.random.choice(len(vst_indices), n_points, replace=False)
        for i in range(len(random_index)):
            curr = random_index[i]
            vst_roi_random_point.append([vst_indices[curr][0], vst_indices[curr][1]])
            if mode == 'inc':
                vst_random_point.append(
                    [vst_roi_random_point[i][0] + xmin - 10, vst_roi_random_point[i][1] + ymin - 10])
            else:
                vst_random_point.append(
                    [vst_roi_random_point[i][0] + xmin2, vst_roi_random_point[i][1] + ymin2])
    else:
        if len(vst_indices) < n_points:
            n_points = len(vst_indices)

        random_index = np.random.choice(len(vst_indices), n_points, replace=False)[0]
        vst_roi_random_point.append([vst_indices[random_index][0], vst_indices[random_index][1]])
        if mode == 'inc':
            vst_random_point.append([vst_roi_random_point[0][0] + xmin - 10, vst_roi_random_point[0][1] + ymin - 10])
        else:
            vst_random_point.append(
                [vst_roi_random_point[0][0] + xmin2, vst_roi_random_point[0][1] + ymin2])


    return torch.Tensor(vst_random_point), vst_mask

def VST(img_npy):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
    parser.add_argument('--train_steps', default=60000, type=int, help='total training steps')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=11, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=30000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=45000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--save_model_dir', default='/home/zoe/point_sampling/pretrained/', type=str, help='save model path') # Change
    parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='RGB_VST', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

    args = parser.parse_args()

    # define model
    cudnn.benchmark = True
    net = ImageDepthNet(args)
    net.cuda()
    net.eval()
    # load model (multi-gpu)
    model_path = args.save_model_dir + 'RGB_VST.pth'
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module'
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    vst_mask = VST_test_img(net, img_npy)
    return vst_mask

def VST_test_img(net, img_npy):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(img_npy).convert('RGB')
    image_w, image_h = int(image.size[0]), int(image.size[1])
    image = transform(image)
    images = image.unsqueeze(0)
    images = images.cuda()

    outputs_saliency, outputs_contour = net(images)

    mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency

    image_w, image_h = int(image_w), int(image_h)

    output_s = F.sigmoid(mask_1_1)
    output_s = output_s.data.cpu().squeeze(0)
    new_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_w, image_h))
    ])
    final = new_transform(output_s)

    return np.array(final)
