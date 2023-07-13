#!/home/syu/anaconda3/envs/wildfire/bin/python
# -*- coding: utf-8 -*-

"""
Overview:
    Ensemble submission:
        Positive:
            - Post Model
            - Change (Post - Pre) Model
        Supression:
            - Cloud Model

Usage:
    $ python 027_ensemble_submit.py > inference.log 2> inference.log

"""

import os
import warnings
import random
from pprint import pprint
import copy
import glob
import json
import csv
# import dataclasses
from joblib import Parallel, delayed
from typing import List, Set, Dict, Any
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
from box import Box, from_file
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.optimize import minimize
from sklearn.metrics import jaccard_score
import cv2
import seaborn as sns

import torch
print(torch.__version__)
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm import create_model
from torchmetrics.functional.classification import binary_fbeta_score
import segmentation_models_pytorch as smp

import pytorch_lightning as pl
print(pl.__version__)
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import LightningDataModule, LightningModule

warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)
pd.options.display.max_colwidth = 250
pd.options.display.max_rows = 30
pd.options.display.max_columns = 30


"""
Configuration
"""

class CFG(object):
    # basic
    debug: bool = False
    debug_sample: int = 10
    folds: int  = 4
    seed: int   = 417
    eps: float  = 1e-12
    
    SINGLE_FOLD = False

    # data
    IDX: int = 3
    PATH_SUBMIT_DATA: str = f'../data/california_{IDX}.hdf5'
    
    # models
    PATH_MODEL_ROOT_CHANGE = f'output/V14/A100_V14_CALFalseDEFFalseNONTrue_hrnet_w30BC16TH0.3_R-RPM_R-RPM_fl-tr_WD0.001E15GA1CE-W1UP4BETA0.25R0.5MIX0.5FREE12B32_Scr'
    PATH_MODEL_ROOT_POST   = f'output/V15/A100_V15_CALFalseDEFFalseNONTrue_hrnet_w30BC16TH0.3_R-RPM_fl-tr_0.0001_E15CE-W1UP4BETA0.5R0.1MIX0.5FREE12B32_Scr'
    PATH_MODEL_ROOT_CLOUD  = f'output/cloud/V2/V100_V2_C_resnet34BC4TH0.4_LR0.01T0100MIX0.5B16_fl-tr'
    
    TH_FIRE = 0.3
    TH_CLOUD = 0.75
    
    outdir = f'output/inference/027/'
    os.makedirs(outdir, exist_ok=True)
    
    # public
    fold:int = 0
    
    preprocess: Dict = {
        "input_size": 512,
    }
    

    if debug:
        epoch = 2
        group = 'DEBUG_post'


# box
sub_cfg = Box({k:v for k, v in dict(vars(CFG)).items() if '__' not in k})
    
# 乱数のシードを設定
seed_everything(sub_cfg.seed)
torch.manual_seed(sub_cfg.seed)
np.random.seed(sub_cfg.seed)
random.seed(sub_cfg.seed)
    
pprint(sub_cfg)

# augmentation
tf_dict = {
    'val': A.Compose(
        [
            A.Resize(sub_cfg.preprocess.input_size, sub_cfg.preprocess.input_size),
            ToTensorV2(),
        ]
    ),
}

sub_cfg.augmentation = str(tf_dict).replace('\n', '').replace(' ', '')
pprint(sub_cfg.augmentation)

# models
from utils.models import TimmUnet

class FHalfWithLogitsLoss(torch.nn.Module):
    """ Fbeta Extended  with logits Loss """
    def __init__(self, eps=sub_cfg.eps, beta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta

    def forward(self, y_pr, y_gt):
        return 1 - binary_fbeta_score(torch.sigmoid(y_pr), y_gt, beta=self.beta)

class BCEDiceLoss(torch.nn.Module):
    """ Combine Loss """
    def __init__(self, raito=0.5, beta=0.5):
        super(BCEDiceLoss, self).__init__()
        assert 0 <= raito <= 1, "loss raito invalid."
        
        self.raito = raito
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.BCE_POS_WEIGHT).cuda())
        self.dice_criterion = FHalfWithLogitsLoss(beta=beta)
        
    def forward(self, y_pr, y_gt):
        loss = self.raito * self.bce_criterion(y_pr, y_gt) + (1 - self.raito) * self.dice_criterion(y_pr, y_gt)
        return loss

class FirePostModel(LightningModule):
    def __init__(self, cfg, model_state=''):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        self.model_state = model_state
        self._criterion = eval(cfg.model.loss)
        
    def __build_model(self):
        in_channels = 12
        self.backbone = TimmUnet(
            in_chans=in_channels,
            **self.cfg.model.timmunet.encoder_params,
            )

    def forward(self, x):
        feat = self.backbone(x)
        return feat.squeeze(1)
    
class FireCgangeModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()

        self._criterion = eval(cfg.model.loss)
        
    def __build_model(self):

        in_channels = 24
        self.backbone = TimmUnet(
            in_chans=in_channels,
            **self.cfg.model.timmunet.encoder_params,
            )

    def forward(self, x):
        feat = self.backbone(x)
        return feat.squeeze(1)
    
class CloudModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        
    def __build_model(self):
        in_channels = 4
        self.backbone = TimmUnet(
            in_chans=in_channels,
            **self.cfg.model.timmunet.encoder_params,
            )

    def forward(self, x):
        feat = self.backbone(x)
        return feat.squeeze(1)
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'### MODEL LOAD ###\n### DEVICE: {device} ###')

models_change = []
models_post   = []
models_cloud  = []

# output
save_preds_dir = f'{sub_cfg.outdir}/visualize/'
os.makedirs(save_preds_dir, exist_ok=True)

for fold in range(sub_cfg.folds):
    PATH_MODEL_ROOT = f'{sub_cfg.PATH_MODEL_ROOT_CHANGE}/A100_fold{fold}'
    PATH_MODEL = f'{PATH_MODEL_ROOT}/iou_fold{fold}.ckpt'
    PATH_CFG   = f'{PATH_MODEL_ROOT}/cfg.json'
    print(PATH_MODEL, PATH_CFG, sep='\n')
    
    # model
    cfg = from_file.box_from_file(PATH_CFG)
    model = FireCgangeModel.load_from_checkpoint(PATH_MODEL, cfg=cfg)
    model.eval()
    model = model.to(device)
    models_change.append(model)
    
for fold in range(sub_cfg.folds):
    PATH_MODEL_ROOT = f'{sub_cfg.PATH_MODEL_ROOT_POST}/A100_fold{fold}'
    PATH_MODEL = f'{PATH_MODEL_ROOT}/iou_fold{fold}.ckpt'
    PATH_CFG   = f'{PATH_MODEL_ROOT}/cfg.json'
    print(PATH_MODEL, PATH_CFG, sep='\n')

    # model
    cfg = from_file.box_from_file(PATH_CFG)
    model = FirePostModel.load_from_checkpoint(PATH_MODEL, cfg=cfg)
    model.eval()
    model = model.to(device)
    models_post.append(model)

for fold in range(sub_cfg.folds):
    PATH_MODEL_ROOT = f'{sub_cfg.PATH_MODEL_ROOT_CLOUD}/V100_fold{fold}'
    PATH_MODEL = f'{PATH_MODEL_ROOT}/iou_fold{fold}.ckpt'
    PATH_CFG   = f'{PATH_MODEL_ROOT}/cfg.json'
    print(PATH_MODEL, PATH_CFG, sep='\n')
    
    # model
    cfg = from_file.box_from_file(PATH_CFG)
    model = CloudModel.load_from_checkpoint(PATH_MODEL, cfg=cfg)
    model.eval()
    model = model.to(device)
    models_cloud.append(model)


## metrics
uuid_s = []
fp_s   = []
p_imgs = []

with h5py.File(sub_cfg.PATH_SUBMIT_DATA, 'r') as fp:
    for num_idx, (uuid, values) in enumerate(fp.items()):
        
        print(f'### [{num_idx}/{len(fp)}] {uuid}')
        uuid_s.append(uuid)
        
        # get data
        post = values['post_fire'][...]
        
        if "pre_fire" not in values:
            IS_DEFECT = True
            
            pre = np.zeros(post.shape, dtype=post.dtype)
        else:
            IS_DEFECT = False
            pre = values['pre_fire'][...]
            
        features = np.concatenate([post, pre], axis=2).astype(np.float32) # H: 512, W:512, C:12
        image = tf_dict['val'](image=features)['image']
        images = image.unsqueeze(dim=0)
        
        # cuda
        images = images.to(device)
        images = images.float()
        
        fires  = []
        clouds = []
        
        # inference
        with torch.no_grad():
            
            if not IS_DEFECT:
                # change
                for model in models_change:
                    outputs = model(images)
                    outputs = torch.sigmoid(outputs)
                    fires.append(outputs[0].cpu().numpy()) # H, W
        
            # post
            for model in models_post:
                outputs = model(images[:, 12:])
                outputs = torch.sigmoid(outputs)
                fires.append(outputs[0].cpu().numpy())
                
            # cloud
            for model in models_cloud:
                
                if not IS_DEFECT:
                    inputs_post = torch.stack([images[:, 1], images[:, 2], images[:, 3], images[:, 7]], dim=1)
                    inputs_pre = torch.stack([images[:, 1+12], images[:, 2+12], images[:, 3+12], images[:, 7+12]], dim=1)
                    
                    # boolean pulse
                    outputs = ((torch.sigmoid(model(inputs_post)) > sub_cfg.TH_CLOUD) + (torch.sigmoid(model(inputs_pre)) > sub_cfg.TH_CLOUD)).float()
                else:
                    inputs = torch.stack([images[:, 1], images[:, 2], images[:, 3], images[:, 7]], dim=1)
                    outputs = ((torch.sigmoid(model(inputs)) > sub_cfg.TH_CLOUD)).float()
                    
                clouds.append(outputs[0].cpu().numpy())
        
        # ensemble
        fire = np.stack(fires, axis=0).mean(axis=0) # C:ensemble, H, W -> H, W
        cloud = np.stack(clouds, axis=0).mean(axis=0) # C:ensemble, H, W -> H, W    
        
        # supression
        fire_bin = (fire > sub_cfg.TH_FIRE).astype(np.float32)
        cloud_bin = (cloud > 0.5).astype(np.float32)
        fire_bin_sup = fire_bin * (1 - cloud_bin)
        
        fp_s.append(fire_bin_sup.sum())
        
        image = (np.clip(image[2:5].permute(1, 2, 0).cpu().numpy(), a_min=1, a_max=5000)/5000*255).astype(np.uint8)
        
        # save
        PATH_IMG = f'{save_preds_dir}/{uuid}.png'
        p_imgs.append(PATH_IMG)
        cv2.imwrite(PATH_IMG, image)
        PATH_FIRE = f'{save_preds_dir}/{uuid}_fire.png'
        cv2.imwrite(PATH_FIRE, (fire*255).astype(np.uint8))
        PATH_CLOUD = f'{save_preds_dir}/{uuid}_cloud.png'
        cv2.imwrite(PATH_CLOUD, (cloud*255).astype(np.uint8))
        PATH_FIRE_BIN = f'{save_preds_dir}/{uuid}_fire_bin.png'
        cv2.imwrite(PATH_FIRE_BIN, (fire_bin*255).astype(np.uint8))
        PATH_FIRE_SUP = f'{save_preds_dir}/{uuid}_fire_sup.png'
        cv2.imwrite(PATH_FIRE_SUP, (fire_bin_sup*255).astype(np.uint8))
        
        
        
        if sub_cfg.debug and num_idx > sub_cfg.debug_sample:
            print(values.keys(), uuid)
            break

df = pd.DataFrame({'uuid': uuid, 'fp_s': fp_s, 'p_img': p_imgs})
df.to_csv(f'{sub_cfg.outdir}/metrics_idx{sub_cfg.IDX}.csv', index=False)

print(f'### Done -> {save_preds_dir}')

