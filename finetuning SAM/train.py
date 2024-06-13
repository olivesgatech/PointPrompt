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
parser.add_argument('--train_ratio', type=float, help='training dataset ratio (e.g. 0.8)')
parser.add_argument('--path_to_sam', type=str, help='the path to huge SAM weights')
args = parser.parse_args()
test_filenames = np.load('test.npy')

# Process filenames to exclude everything before "Image datasets"
prnames = [ os.path.normpath(filename.split('samples/')[1:] [0]) for filename in test_filenames]


methods = [
   'human' ,
   'rand',
    'saliency',
   'kmediod',
    'entropy',
    'max_dist',
  'shi-Tomasi']
for method in methods:
    print(method)
    model = sam_model_registry["vit_h"](checkpoint=args.path_to_sam)
    optimizer = Adam(model.prompt_encoder.parameters(), lr=1e-3, weight_decay=0)

    loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    num_epochs = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    losses=[]
    model.train()
    nimprove=0
    losses=[]
    for name, param in model.named_parameters():
      if name.startswith("image_encoder") or name.startswith("mask_decoder"):
        param.requires_grad_(False)
    #model = torch.nn.DataParallel(model,device_ids=[0,1])
    model = model.to(device)
    if not method=='human':
        trdataset = SAMDataset(args.path,method,'train', processor=processor)
    else: 
        trdataset = SAMDataset_humans(args.path,method, 'train', processor=processor)
    # to check if the training set is loaded correctly:
    print(trdataset.samples[:10])
    print(trdataset.labels[:10])
    print(trdataset.gpoints[:10])
    train_size = int(args.train_ratio * len(trdataset))
    valid_size = len(trdataset) - train_size
    torch.manual_seed(42)
    # Randomly split the dataset
    train_dataset, valid_dataset = random_split(trdataset, [train_size, valid_size])
    
    # loader with batch size one. there is an error if you set the batch size larger than 1. 
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_losses = []
        b=0
        if nimprove > 4:
            print("no imrpove on last 5 epochs")
            break
        for batch in tqdm(train_dataloader):
            b+=1

            pixel_values=batch["pixel_values"]
            prompts=batch["input_labels"]
            #image embeddings 
            img_embs=model.image_encoder(pixel_values)
    
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(batch["input_points"].squeeze(0),batch["input_labels"].squeeze(0)),
            boxes=None,
            #masks=mask[:][0],
            masks=None
            )
            low_res_masks,iou=model.mask_decoder(image_embeddings=img_embs,
                                              image_pe = model.prompt_encoder.get_dense_pe(),
                                              sparse_prompt_embeddings=sparse_embeddings,
                                              dense_prompt_embeddings=dense_embeddings, multimask_output=False  
                                              )
            
            ground_truth_masks = batch['ground_truth_mask'].float().to(device).unsqueeze(1)
            
            loss = loss_fn(low_res_masks,ground_truth_masks)
            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()
    
            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())
            
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
        losses.append(mean(epoch_losses))
        # Validation
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch in valid_dataloader:
                pixel_values = batch["pixel_values"]
                prompts = batch["input_labels"]
    
                img_embs = model.image_encoder(pixel_values)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=(batch["input_points"].squeeze(0), batch["input_labels"].squeeze(0)),
                    boxes=None,
                    masks=None
                )
    
                low_res_masks, iou = model.mask_decoder(
                    image_embeddings=img_embs,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
    
                ground_truth_masks = batch['ground_truth_mask'].float().to(device).unsqueeze(1)
                loss = loss_fn(low_res_masks, ground_truth_masks)
    
                valid_losses.append(loss.item())
    
        valid_loss = np.mean(valid_losses)
        print(f'Validation loss: {valid_loss}')
        # save the model only if the validation loss is improved w.r.t the best validation loss in the past iterations
        if np.round(valid_loss,3) < np.round(best_valid_loss,3):
            print("Valid loss",valid_loss)
            print("Best Valid Loss so far:",best_valid_loss)
            best_valid_loss = valid_loss
            
            torch.save(model.state_dict(), f'{method}.pth')
            print(f'Model saved with validation loss: {best_valid_loss}')
            
        # if there is not improvment in 5 iterations then stop training.
        else:
            nimprove += 1
        #break
        if epoch>1 and np.round(losses[-1],3)>=np.round(losses[-2],3):
            nimprove+=1
