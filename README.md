# Abstract
The remarkable capabilities of the Segment Anything Model (SAM) for tackling image segmentation tasks in an intuitive and interactive manner has sparked interest in the design of effective visual prompts. Such interest has led to the creation of automated point prompt selection strategies, typically motivated from a feature extraction perspective. However, there is still very little understanding of how appropriate these automated visual prompting strategies are, particularly when compared to humans, across diverse image domains. Additionally, the performance benefits of including such automated visual prompting strategies within the finetuning process of SAM also remains unexplored, as does the effect of interpretable factors like distance between the prompt points on segmentation performance. To bridge these gaps, we leverage a recently released visual prompting dataset, PointPrompt, and introduce a number of benchmarking tasks that provide an array of opportunities to improve the understanding of the way human prompts differ from automated ones and what underlying factors make for effective visual prompts. We demonstrate that the resulting
segmentation scores obtained by humans are approximately 29% higher than those given by automated strategies and identify potential features that are indicative of prompting performance with R2 scores over 0.5. Additionally, we demonstrate that performance when using automated methods can be improved by up to 68% via a finetuning approach. Overall, our experiments not only showcase the existing gap between human prompts and automated methods, but also highlight potential avenues through which this gap can be leveraged to improve effective visual prompt design. Further details along with the dataset links and codes are available at [this link](https://alregib.ece.gatech.edu/pointprompt-a-visual-prompting-dataset-based-on-the-segment-anything-model/).

# Dataset
Access the **dataset** [here](https://zenodo.org/records/11187949).
This dataset has two zip files: **Image datasets.zip** and **Prompting data.zip**.

**Image datasets.zip** contains all image datasets, along with ground truth labels. For each image dataset, there are 400 image-ground truth mask pairs. The image and ground truth masks are formatted as .npy arrays.
**Prompting data.zip** contains prompting data collected from human annotators. The structure appears as the following:

```
Prompting Results
├── Baseball bat                                 # Image dataset
    ├── st1                                      # Human annotator # 1
        ├── eachround                            # List of length t (number of timestamps); indicates which timesteps belong to each of the two rounds (if they exist)
        ├── masks                                # Contains the binary masks produced for each image, in format a_b_mask.png, where 'a' corresponds to the image number (0 to 399) and 'b' indexes through timestamps in the prompting process
        ├── points                               # Contains inclusion and exclusion points formatted as a_green.npy and a_red.npy respectively, where 'a' corresponds to the image number
        ├── scores                               # Contains the scores at each timestep for every image (mIoU)
        ├── sorts                                # Contains sorted timestamp indexes, going from max to min based on the score
    ├── st2                                      # Human annotator # 2 (same structure as st1)
        .
        .
        .
    ├── st3                                      # Human annotator # 3 (same structure as st1)
        .
        .
        .
.
.
.
├── Tie
```

# Code Usage

Please download the weights for SAM via:

```
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth <br>
```

or from this [direct-link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and place it in your repository folder. 

**Point Sampling Strategy Experiments:**
1. Go to the `point_sampling` directory
2. Make sure you read `Instructions.md`. If you wish to run the `Saliency` strategy, follow instructions to download the appropriate items.
3. The main script (``python3 main.py``) runs the code. You will need to alter `--img_dir` to the folder which keeps the image datasets (download above). You will also need to adjust `--results_dir` to where you wish for the results to be saved. Additionally, `--home_dir` should be the directory where the pretrained weights are located within the repository.
4. You can specify which strategy you wish to run by altering the `--query_strategy` parameter.

**Finetuning:**
For finetuning the structure of the dataset should be **altered** to look this way: 
```
Image datasets        # Image dataset
├── human             # the first prompting strategy. the human is a special case since it contains multiple prompt sets for each image
    ├── Baseball bat  # the dataset type            
        ├── st1_0_green.npy # the inclusion points for first user (st1) done on the first image (_0_) 
        ├── st1_0_red.npy # the exclusion points for first user (st1) done on the first image (_0_)
        ├── st1_1_green.npy # the inclusion points for first user (st1) done on the second image (_1_) 
        ├── st1_1_red.npy # the exclusion points for first user (st1) done on the second image (_1_) 
        .
        .
        .
        ├── st3_399_green.npy # the inclusion points for third user (st3) done on the 399 image (_399_) 
        ├── st3_399_red.npy # the exclusion points for third user (st3) done on the 399 image (_399_) 
    ├── Tie
├── entropy # the other prompting strategies do no have multiple prompting sets for each image.
    ├── Baseball bat  # the dataset type            
        ├── 0_green.npy # the inclusion points done on the first image (_0_) 
        ├── 0_red.npy # the exclusion points done on the first image (_0_)
        .
        .
        .
        ├── 399_green.npy # the inclusion points done on the 399 image (_399_) 
        ├── 399_red.npy # the exclusion points done on the 399 image (_399_) 
    ├── Tie
.
.
.
├── samples
    ├── Baseball bat  # the dataset type            
        ├── 0_sample.npy # the first image in the Baseball bat dataset 
        ├── 1_sample.npy # the second image in the Baseball bat dataset 
        .
        .
        .
        ├── 399_sample.npy # the 399th image in the Baseball bat dataset
    ├── Tie
├── labels
    ├── Baseball bat  # the dataset type            
        ├── 0_label.npy # the first ground truth mask in the Baseball bat dataset 
        ├── 1_label.npy # the second ground truth mask in the Baseball bat dataset 
        .
        .
        .
        ├── 399_label.npy # the 399th ground truth mask in the Baseball bat dataset
    ├── Tie
```
So the Image datasets folder contain the prompting strategies files, also the images files under samples, and labels (ground truth) file.

1- Download the requirements: 
```
pip install -r requirements.txt
```
2- To finetune the prompt encoder based on each prompting method. The weights will be saved by the strategy name: 
```
python train.py --path "Image datasets" --train_ratio 0.8 --path_to_sam "sam_vit_h_4b8939.pth" 
```
3- Test each model on all data and save the results in an excel sheet.
```
python inference.py --path "Image datasets" 

```

**Feature extraction:**
To extract the image, prompt and general-level features described in the paper, first make sure you have downloaded both the image and the prompt data and stored it into corresponding folders. First download the required packages:
```
pip install -r requirements.txt
```
and then run the feature extractions script providing the paths where the data is stored:
```
python gather_statistics.py --data_path "\path\to\images" --prompts_path "\path\to\prompts"
```

# Links

Automated point selection strategies:

[SAMAug](https://github.com/yhydhx/SAMAug)

[SAM-PT](https://github.com/SysCV/sam-pt)

# Citation

J. Quesada∗, Z. Fowler∗, M. Alotaibi, M. Prabhushankar, and G. AlRegib, ”Benchmarking Human and
Automated Prompting in the Segment Anything Model”, In IEEE International Conference on Big Data
2024, Washington DC, USA
