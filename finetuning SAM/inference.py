# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 20:25:25 2024

@author: Mohammed
"""

import torch.nn as nn
from scipy.ndimage import zoom
from transformers import SamModel 
import torch
import json
import os
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from transformers import SamProcessor 
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
from torch.optim import Adam
import monai
import pandas as pd 
import cv2
from PIL import Image
import argparse
#preferably GPU otherwise it will take time. 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load huge SAM processor, used for data pre-processing. You can use base SAM or even large SAM, but huge SAM is preferable for better results
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

#this function to pad and the labels (ground truth) and resize them to SAM's output size (256*256)
def scale_and_pad_label(image, new_max_length=256):
    # Calculate the scaling factor
    height, width = np.shape(image)
    scaling_factor = new_max_length / max(height, width)
    image = image.astype(np.uint8) 
    # Scale the image
    new_height = int(height * scaling_factor)
    new_width = int(width * scaling_factor)
    #scaled_image = resize(image, (new_height, new_width), preserve_range=True, anti_aliasing=True).astype(image.dtype)
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Calculate padding
    pad_height = new_max_length - new_height
    pad_width = new_max_length - new_width
    
    # Apply padding to the far left side
    #scaled_image=(scaled_image==5)*1
    padded_image = np.pad(scaled_image, ((0,pad_height), 
                                          (0,pad_width)), 
                          'constant', constant_values=0)
    return padded_image
# sam dataset class for automatic prompting methods: 
class SAMDataset_humans(Dataset):
  def __init__(self, path,method,subset, processor):
    clss= os.listdir(os.path.join(path,"samples")) # get image type 
    self.samples=[]
    self.labels=[]
    self.gpoints=[]
    self.rpoints=[]
    #loop through all image datasets:
    for cl in clss: 
        if cl == 'Pizza':
            continue 
        ranges=len(os.listdir(os.path.join(path,"samples",cl)))
        #load the training set
        if subset=='train':
            
            # loop through all the images
            for i in range(ranges):
                # some images are labeled by multiple users st1,st2 , .. st5
                for st in range(1,5):
                        if os.path.exists(  os.path.join(path,method,cl,f'st{st}_{i}_green.npy')  ) and  not os.path.join(cl,f'{i}_sample.npy') in prnames:
                            self.samples.append( os.path.join(path,"samples",cl,f'{i}_sample.npy') )
                            self.labels.append( os.path.join(path,"labels",cl,f'{i}_label.npy') )
                            self.gpoints.append(os.path.join(path,method,cl,f'st{st}_{i}_green.npy') )
                            self.rpoints.append(os.path.join(path,method,cl,f'st{st}_{i}_red.npy') )


        # load the testing set                    
        else: 
            for i in range(ranges):
                # some images are labeled by multiple users st1,st2 , .. st5
                for st in range(1,5):
                        if os.path.exists(  os.path.join(path,method,cl,f'st{st}_{i}_green.npy')  ) and  os.path.join(cl,f'{i}_sample.npy') in prnames:
                            self.samples.append( os.path.join(path,"samples",cl,f'{i}_sample.npy') )
                            self.labels.append( os.path.join(path,"labels",cl,f'{i}_label.npy') )
                            self.gpoints.append(os.path.join(path,method,cl,f'st{st}_{i}_green.npy') )
                            self.rpoints.append(os.path.join(path,method,cl,f'st{st}_{i}_red.npy') )

    self.processor = processor

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = np.load(self.samples[idx])
    ground_truth_mask = np.load(self.labels[idx])
    gr=np.load(self.gpoints[idx],allow_pickle=True) 
    rd= np.load(self.rpoints[idx],allow_pickle=True)
    input_point = np.concatenate((gr, rd))
    input_label = np.concatenate(([1] * len(gr), [0] * len(rd)))
    
    # for the seismic images, normalize the image to be 0-255 and convert them to RGB:
    if np.min(image)==-1:
        image=(((image+1)/2)*255).astype(np.uint8)
        image=np.dstack((image,image,image))
    # prepare image and prompt for the model.sha
    inputs = self.processor(image, input_points= [[input_point]], input_labels=[[input_label]], return_tensors="pt").to(device)
    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}
    # add ground truth segmentation
    inputs["ground_truth_mask"] = scale_and_pad_label(ground_truth_mask)
    
    return inputs
    
class SAMDataset(Dataset):
  def __init__(self, path,method,subset, processor):
    clss= os.listdir(os.path.join(path,"samples")) # get image type 
    self.samples=[]
    self.labels=[]
    self.gpoints=[]
    self.rpoints=[]
    #loop through all image datasets except for pizza because it does not have labels:
    for cl in clss: 
        if cl == 'Pizza':
            continue 
        ranges=len(os.listdir(os.path.join(path,"samples",cl)))
        # load the trianing set
        if subset=='train':
            print(cl)
            for i in range(ranges):
                    #unlike human prompts, auto prompts have only one set of prompts for each image. so no need to loop through multiple users
                    if os.path.exists(  os.path.join(path,method,cl,f'{i}_green.npy')  ) and  not os.path.join(cl,f'{i}_sample.npy') in prnames:
                        self.samples.append( os.path.join(path,"samples",cl,f'{i}_sample.npy') )
                        self.labels.append( os.path.join(path,"labels",cl,f'{i}_label.npy') )
                        self.gpoints.append(os.path.join(path,method,cl,f'{i}_green.npy') )
                        self.rpoints.append(os.path.join(path,method,cl,f'{i}_red.npy') )

        # testing set                    
        else: 
            for i in range(ranges):
                if os.path.exists(  os.path.join(path,method,cl,f'{i}_green.npy')  ) and  os.path.join(cl,f'{i}_sample.npy') in prnames:
                    self.samples.append( os.path.join(path,"samples",cl,f'{i}_sample.npy') )
                    self.labels.append( os.path.join(path,"labels",cl,f'{i}_label.npy') )
                    self.gpoints.append(os.path.join(path,method,cl,f'{i}_green.npy') )
                    self.rpoints.append(os.path.join(path,method,cl,f'{i}_red.npy') )

    self.processor = processor

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = np.load(self.samples[idx])
    ground_truth_mask = np.load(self.labels[idx])
    gr=np.load(self.gpoints[idx],allow_pickle=True) 
    rd= np.load(self.rpoints[idx],allow_pickle=True)
    input_point = np.concatenate((gr, rd))
    input_label = np.concatenate(([1] * len(gr), [0] * len(rd)))
    
    # for the seismic images, normalize the image to be 0-255 and convert them to RGB:
    if np.min(image)==-1:
        image=(((image+1)/2)*255).astype(np.uint8)
        image=np.dstack((image,image,image))
    # prepare image and prompt for the model.sha
    inputs = self.processor(image, input_points= [[input_point]], input_labels=[[input_label]], return_tensors="pt").to(device)
    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}
    # add ground truth segmentation
    inputs["ground_truth_mask"] = scale_and_pad_label(ground_truth_mask)
    
    return inputs
parser = argparse.ArgumentParser(description='Settings')    
parser.add_argument('--path', type=str, help='The path to the prompt datasets')
args = parser.parse_args()
test_filenames = np.load('test.npy')


# Function to evaluate a model on a dataset
def evaluate_model(testdataset, model):
    model = model.to(device=torch.device("cuda"))
    predictor = SamPredictor(model)
    model.eval()
    
    miou = []
    with torch.no_grad():
        for batch in testdataset:
            
            pixel_values = batch["pixel_values"]
            prompts = batch["input_labels"]
            label = batch["ground_truth_mask"]
            predictor.set_image(pixel_values)
            masks, scores, logits = predictor.predict(
                point_coords=batch['input_points'],
                point_labels=batch['input_labels'],
                multimask_output=False,
            )
            intersection = (masks[0] & label.astype(bool)).sum()
            union = (masks[0] | label.astype(bool)).sum()
            if intersection == 0:
                s = 0
            else:
                s = intersection / union
            miou.append(s)
    return np.mean(miou), np.std(miou).item()

# Function to evaluate human model
def evaluate_human_model(testdataset):
    miou = []
    for batch in testdataset:
        label = batch["ground_truth_mask"]
        human_mask = batch["human_mask"]
        intersection = (human_mask & label).sum()
        union = (human_mask | label).sum()
        if intersection == 0:
            s = 0
        else:
            s = intersection / union
        miou.append(s)
    return np.mean(miou), np.std(miou).item()

# Process filenames to exclude everything before "Image datasets"
prnames = [ os.path.normpath(filename.split('samples/')[1:] [0]) for filename in test_filenames]

columns = ['Category'] + [f'{key} mIOU' for key in models]
models = [
   'human' ,
   'rand',
    'saliency',
   'kmediod',
    'entropy',
    'max_dist',
  'shi-Tomasi']

# Create an empty DataFrame with the required columns
results_df = pd.DataFrame(columns=columns)
results_df['Category'] = datasets

# Iterate over models and datasets, and store the results
for m in models: 
    print(m)
    model = sam_model_registry["vit_h"](checkpoint=f'{m.lower()}.pth')
    for model_name in models:
        print(model_name)
        if model_name.lower() == 'human':
            testdataset = SAMDataset_humans(path, model_name.lower(), 'test', processor) 
        else: 
            testdataset = SAMDataset(path, model_name.lower(), 'test', processor)
        print(len(testdataset))
        mean_iou, std_iou = evaluate_model(testdataset, model)
        results_df.loc[results_df['Category'] == dataset, f'{model_name} mIOU'] = f'{mean_iou:.3f} ± {std_iou:.3f}'

    # Save the DataFrame to an Excel file
results_df.to_excel(f'{m}.xlsx', index=False)
