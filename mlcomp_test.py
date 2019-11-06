import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt

import numpy as np
import cv2
import albumentations as A
from torchvision.transforms import Resize
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.jit import load

from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import rle2mask, mask2rle
from mlcomp.contrib.transform.tta import TtaWrap
from config import *
import segmentation_models_pytorch as smp
from PIL import Image

def get_black_mask(image_path):
    img = cv2.imread(image_path)
    #img = cv2.resize(img, (525,350))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([180, 255, 10], np.uint8)
    return (~ (cv2.inRange(hsv, lower, upper) > 250)).astype(int)


def sharpen_mean(prob, dim = 0):
    p = prob ** 0.5
    p = torch.mean(p, dim)
    return p

class Model:
    def __init__(self, models):
        self.models = models
    
    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append((m(x)))
        res = torch.stack(res)
        return torch.mean(res, dim=0)

def create_transforms(additional):
    res = list(additional)
    # add necessary transformations
    res.extend([
    	#Resize((320, 512), interpolation = Image.INTER_NEAREST),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
        ChannelTranspose()
    ])
    res = A.Compose(res)
    return res

img_folder = params['test_path']
batch_size = 2
num_workers = 0

UnetPlus = ResNet34UnetPlus(num_class = 4)
checkpoint = torch.load(params['best_acc'], map_location='cpu')
UnetPlus = nn.DataParallel(UnetPlus).cuda()
UnetPlus.load_state_dict(checkpoint)
'''
ENCODER = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
Unet = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=4, 
    activation=None,
    )
checkpoint = torch.load(params['best_acc'], map_location='cpu')
Unet = nn.DataParallel(Unet).cuda()
Unet.load_state_dict(checkpoint)
'''
model = Model([UnetPlus])

# Different transforms for TTA wrapper
transforms = [
    [],
    [A.HorizontalFlip(p=1)]
]

transforms = [create_transforms(t) for t in transforms]
datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]

loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]


thresholds = [0.5, 0.5, 0.5, 0.5]
min_area = [30000, 10000, 30000, 10000]
count = 0
res = []
cls_dict = ['Fish', 'Flower', 'Gravel', 'Sugar']
# Iterate over all TTA loaders
total = len(datasets[0])//batch_size
for loaders_batch in tqdm(zip(*loaders), total=total):
    preds = []
    image_file = []
    classify = []
    for i, batch in enumerate(loaders_batch):
        features = batch['features'].cuda()
        features = F.interpolate(features, (320,512), mode = 'nearest')
        p = torch.sigmoid(model(features))
        p = F.interpolate(p, (350, 525), mode = 'nearest')
        # inverse operations for TTA
        p = datasets[i].inverse(p)
        preds.append(p)
        image_file = batch['image_file']

    preds = torch.stack(preds)
    preds = torch.mean(preds, dim=0)
    preds = preds.detach().cpu().numpy()
    for p in preds:
        save_mask = np.argmax(p , axis = 0)
        count += 1
        cv2.imwrite(params['output'] + str(count)+'.png', save_mask * 50)
        
    
    # Batch post processing
    for p, file in zip(preds, image_file):
        file = os.path.basename(file)
        # Image postprocessing
        black_mask = get_black_mask(params['test_path'] + file)
        black_mask = cv2.resize(black_mask, (525, 350), interpolation=cv2.INTER_NEAREST)
        for i in range(4):
            p_channel = p[i] * black_mask
            imageid_classid = file+'_'+cls_dict[i]
            p_channel = (p_channel>thresholds[i]).astype(np.uint8)
            if p_channel.sum() < min_area[i]:
                p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)

            res.append({
                'Image_Label': imageid_classid,
                'EncodedPixels': mask2rle(p_channel)
            })
        
df = pd.DataFrame(res)
df.to_csv(params['work_dir'] + '/submission.csv', index=False)	