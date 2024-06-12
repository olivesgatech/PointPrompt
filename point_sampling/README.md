# Prior to Running this Folder
1. Please download the weights for SAM via:

```
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth <br>
```

or from this [direct-link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and place it in your repository folder. 

2. For running the VST / saliency strategy, please refer to these [instructions](https://github.com/yhydhx/SAMAug/tree/a7dce8878d56d2a265bd2819de3246c726e4adb4/vst_main):

Download the pretrained T2T-ViT_t-14 model [[Google drive](https://drive.google.com/file/d/1R63FUPy0xSybULqpQK6_CTn3QgNog32h/view?usp=sharing)] and put it into a folder called `pretrained`.
Download the pretrained RGB_VST.pth [[Google drive](https://drive.google.com/file/d/1tZ3tQkQ7jlDDfF-_ZROnEZg44MaNQFMc/view)] and put it into a folder called `pretrained`.
