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
    $ python 036_validation_fp_additional.py > fp_validation_add.log 2>  fp_validation_add.log

"""

import os
import datetime
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
from trimesh.voxel.runlength import dense_to_brle

import torch
print(torch.__version__)
import torch.optim as optim
import torch.nn as nn
import ttach as tta
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
    debug: bool       = True
    debug_sample: int = 20
    folds: int        = 4
    seed: int         = 417
    eps: float        = 1e-12
    
    SINGLE_FOLD: bool = False # only 1 fold
    IS_IMG_SAVE: bool = True
    IS_TTA: bool      = True # https://github.com/qubvel/ttach
    NO_CLOUD_MODEL: bool = True
    NO_POST_MODEL: bool = True


    # data
    PATH_CV_DATA:str = f'../data/035_Sydony/019_seed417.csv'
    
    # models
    PATH_MODEL_ROOT_CHANGE_S: List[str] = [
        'output/V6/A6000_V6_CALFalseDEFFalseNONFalse_hrnet_w32BC32TH0.4_LR0.0001T0200BETA0.5MIX0.5B32',
        # f'output/V16/A100_V16_CALFalseDEFTrueNONTrue_hrnet_w30BC16TH0.3_R-RPM_R-RPM_fl-tr_WD1e-05E15GA1CE-W1UP8BETA0.25R0.1MIX0.5FREE12B16_Scr',
        # 'output/V10/3090_V10_CALFalseDEFTrueNONFalse_hrnet_w30BC16TH0.3_R-RPM_R-RPM_fl-tr-gbr_WD0.001E200BETA0.25MIX0.5B16',
    #    'output/V64/A100_V64_CALFalseDEFTrueNONTrue_hrnet_w30BC16TH0.3_R-RPM_R-RPM_fl-tr_WD1e-05E10GA1CE-W1UP16BETA0.25R0.5MIX0.5FREE8B16_Scr',
        # 'output/V18/A100_V18_Sia_CALFalseDEFFalseNONTrue_hrnet_w30BC16TH0.3_R-RPM_R-RPM_fl-trE12GA1CE-W1UP8BETA0.25R0.5MIX0.5FREE9B16_Scr',
        # 'output/V14/A100_V14_CALFalseDEFFalseNONTrue_hrnet_w30BC16TH0.3_R-RPM_R-RPM_fl-tr_WD0.001E15GA1CE-W1UP4BETA0.25R0.5MIX0.5FREE12B32_Scr',
        # 'output/V19/A6000_V19_Sia-POS_CALFalseDEFTrueNONTrue_hrnet_w30BC16TH0.3_R-RPM_R-RPM_fl-tr-gbr_E40CE-W1BETA0.25R0.5MIX0.5FREE32B16_Scr',
    ]
    PATH_MODEL_ROOT_POST_S: List[str]      = [
        # f'output/V17/A100_V17_CALFalseDEFFalseNONTrue_hrnet_w30BC16TH0.3_R-RPM_fl-tr_0.0001_E15CE-W1UP6BETA0.25R0.5MIX0.5FREE12B16_Scr',
        'output/V15/A100_V15_CALFalseDEFFalseNONTrue_hrnet_w30BC16TH0.3_R-RPM_fl-tr_0.0001_E15CE-W1UP4BETA0.5R0.1MIX0.5FREE12B32_Scr'
    ]
    PATH_MODEL_ROOT_CLOUD_S: List[str]     = [
        f'output/cloud/V2/V100_V2_C_resnet34BC4TH0.4_LR0.01T0100MIX0.5B16_fl-tr',
    ]
    
    TH_FIRE:float  = 0.4
    TH_CLOUD:float = 0.8
    
    CLOUD_POST_ONLY:bool = True
    
    outdir:str = f'output/val_fp_cal/036/Sydony_v6-v15_th{TH_FIRE}_cloud-post{CLOUD_POST_ONLY}_no-cloud{NO_CLOUD_MODEL}_no-post{NO_POST_MODEL}/'
    os.makedirs(outdir, exist_ok=True)
    
    # public
    fold:int = 0
    SKIP_THIS_FOLD: bool = False # cv score leak
    
    preprocess: Dict = {
        "input_size": 512,
    }
dt_now = datetime.datetime.now()
dt_now = str(dt_now).replace(' ', '_').replace(':', '_')

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

def compute_submission_mask(id: str, mask):
    """
    official submission format
        refer from: https://huggingface.co/datasets/chabud-team/chabud-ecml-pkdd2023/blob/main/create_sample_submission.py
    """
    brle = dense_to_brle(mask.astype(bool).flatten())
    return {"id": id, "rle_mask": brle, "index": np.arange(len(brle))}

# models
from utils.models import TimmUnet, SiameseFuseUnet

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
        try:
            self._criterion = eval(cfg.model.loss)
        except:
            print('No Loss Weight Model')
        
    def __build_model(self):
        in_channels = 12
        self.backbone = TimmUnet(
            in_chans=in_channels,
            **self.cfg.model.timmunet.encoder_params,
            )

    def forward(self, x):
        feat = self.backbone(x)
        return feat
    
class FireChangeModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()

        try:
            self._criterion = eval(cfg.model.loss)
        except:
            print('No Loss Weight Model')
        
    def __build_model(self):

        in_channels = 24
        self.backbone = TimmUnet(
            in_chans=in_channels,
            **self.cfg.model.timmunet.encoder_params,
            )

    def forward(self, x):
        feat = self.backbone(x)
        return feat
    

class FireSiameseModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()

        try:
            self._criterion = eval(cfg.model.loss)
        except:
            print('No Loss Weight Model')
        
    def __build_model(self):

        in_channels = 24
        self.backbone = SiameseFuseUnet(
            in_chans=in_channels,
            **self.cfg.model.timmunet.encoder_params,
            )

    def forward(self, x):
        feat = self.backbone(x)
        return feat
    
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
        return feat
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'### MODEL LOAD ###\n### DEVICE: {device} ###')

models_change = []
models_post   = []
models_cloud  = []

# output
save_preds_dir = f'{sub_cfg.outdir}visualize/'
os.makedirs(save_preds_dir, exist_ok=True)

for _PATH_ROOT in sub_cfg.PATH_MODEL_ROOT_CHANGE_S:

    for fold in range(sub_cfg.folds):
        
        if sub_cfg.SKIP_THIS_FOLD and fold != sub_cfg.fold:
            continue
        
        if 'A6000' in _PATH_ROOT:
            PATH_MODEL_ROOT = f'{_PATH_ROOT}/A6000_fold{fold}'
        elif '3090' in _PATH_ROOT:
            PATH_MODEL_ROOT = f'{_PATH_ROOT}/3090_fold{fold}'
        else:
            PATH_MODEL_ROOT = f'{_PATH_ROOT}/A100_fold{fold}'

        PATH_MODEL = f'{PATH_MODEL_ROOT}/iou_fold{fold}.ckpt'
        PATH_CFG   = f'{PATH_MODEL_ROOT}/cfg.json'
        print(PATH_MODEL, PATH_CFG, sep='\n')
        
        # model
        cfg = from_file.box_from_file(PATH_CFG)
        
        if 'Sia' in PATH_CFG:
            model = FireSiameseModel.load_from_checkpoint(PATH_MODEL, cfg=cfg)
        else:
            model = FireChangeModel.load_from_checkpoint(PATH_MODEL, cfg=cfg)
            
        model.eval()
        model = model.to(device)
        
        if sub_cfg.IS_TTA:
            model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
        
        models_change.append(model)
        
        if sub_cfg.SINGLE_FOLD:
            break
if not sub_cfg.NO_POST_MODEL:
    for _PATH_ROOT in sub_cfg.PATH_MODEL_ROOT_POST_S:
        _PATH_ROOT = _PATH_ROOT.replace('/A6000_fold', '/A100_fold').replace('/3090_fold', '/A100_fold')
        for fold in range(sub_cfg.folds):
            if sub_cfg.SKIP_THIS_FOLD and fold != sub_cfg.fold:
                continue
            
            if 'A6000' in _PATH_ROOT:
                PATH_MODEL_ROOT = f'{_PATH_ROOT}/A6000_fold{fold}'
            elif '3090' in _PATH_ROOT:
                PATH_MODEL_ROOT = f'{_PATH_ROOT}/3090_fold{fold}'
            else:
                PATH_MODEL_ROOT = f'{_PATH_ROOT}/A100_fold{fold}'
            PATH_MODEL = f'{PATH_MODEL_ROOT}/iou_fold{fold}.ckpt'
            PATH_CFG   = f'{PATH_MODEL_ROOT}/cfg.json'
            print(PATH_MODEL, PATH_CFG, sep='\n')

            # model
            cfg = from_file.box_from_file(PATH_CFG)
            model = FirePostModel.load_from_checkpoint(PATH_MODEL, cfg=cfg)
            model.eval()
            model = model.to(device)
            
            if sub_cfg.IS_TTA:
                model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
                
            models_post.append(model)
            
            if sub_cfg.SINGLE_FOLD:
                break

for _PATH_ROOT in sub_cfg.PATH_MODEL_ROOT_CLOUD_S:
    for fold in range(sub_cfg.folds):
        if sub_cfg.SKIP_THIS_FOLD and fold != sub_cfg.fold:
            continue
        
        if 'A6000' in _PATH_ROOT:
            PATH_MODEL_ROOT = f'{_PATH_ROOT}/A6000_fold{fold}'
        elif '3090' in _PATH_ROOT:
            PATH_MODEL_ROOT = f'{_PATH_ROOT}/3090_fold{fold}'
        else:
            PATH_MODEL_ROOT = f'{_PATH_ROOT}/V100_fold{fold}'
            
        PATH_MODEL = f'{PATH_MODEL_ROOT}/iou_fold{fold}.ckpt'
        PATH_CFG   = f'{PATH_MODEL_ROOT}/cfg.json'
        print(PATH_MODEL, PATH_CFG, sep='\n')
        
        # model
        cfg = from_file.box_from_file(PATH_CFG)
        model = CloudModel.load_from_checkpoint(PATH_MODEL, cfg=cfg)
        model.eval()
        model = model.to(device)
        
        if sub_cfg.IS_TTA:
            model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
            
        models_cloud.append(model)
        
        if sub_cfg.SINGLE_FOLD:
            break
        
# data

class FireDataset(Dataset):
    def __init__(self,
                 df,
                 ):
        self.df = df
        self.transform = tf_dict['val']
        print(f"######## Num sample:", len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        NAME = str(row['name'])
        PATH_FEAT = row.feature
        img_idx = np.load(PATH_FEAT).astype(np.float32)
        PATH_FEAT_SAME = row.same_idx
        img_same = np.load(PATH_FEAT_SAME).astype(np.float32)
        
        img = np.concatenate([img_idx, img_same], axis=2)
        
        # augmentation
        transformed = self.transform(image=img)
        
        img = transformed["image"]
        return img, NAME
    

df = pd.read_csv(sub_cfg.PATH_CV_DATA)
df['h_idx'] = df['name'].apply(lambda x: int(x.split('_')[1][1:]))
df['w_idx'] = df['name'].apply(lambda x: int(x.split('_')[2][1:]))
df['scene'] = df['name'].apply(lambda x: x.split('_')[0])
df['sceen_category'] = df['scene'].factorize()[0]
max_cate = df['sceen_category'].max()

df['search_idx'] = range(df.shape[0])

def search_same_patch_area(row):
    if row.sceen_category == max_cate:
        df_filtered = df[
        (df['h_idx'] == row.h_idx) & (df['w_idx'] == row.w_idx) & (df['sceen_category'] == 0)
        ]
    
    else:
        df_filtered = df[
            (df['h_idx'] == row.h_idx) & (df['w_idx'] == row.w_idx) & (df['sceen_category'] == (row.sceen_category + 1))
            ]
    row_same_area_idx = df_filtered.feature.to_list()
    return row_same_area_idx

tqdm.pandas()
df['same_idx'] = df.progress_apply(search_same_patch_area, axis=1, result_type='expand')

print(df.head())


ds = FireDataset(df=df)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# metrics
uuid_s, fp_s, pos_ratio_s, submits, path_imgs = [], [], [], [], []

# inference
for num_idx, (phase, (images, uuid)) in tqdm(enumerate(zip(['public']*len(ds), dl)), total=len(ds)):
    if isinstance(uuid, list):
        uuid = uuid[0]
    
    # cuda
    images = images.to(device)
    images = images.float()
    
    fires  = []
    clouds = []
    
    # inference
    with torch.no_grad():
        
        # change
        for model in models_change:
            outputs = model(images).squeeze(1)
            outputs = torch.sigmoid(outputs)
            fires.append(outputs[0].cpu().numpy()) # H, W
            
        # post
        for model in models_post:
            outputs = model(images[:, :12]).squeeze(1)
            outputs = torch.sigmoid(outputs)
            fires.append(outputs[0].cpu().numpy())
            
        # cloud
        for model in models_cloud:
            
            if (not sub_cfg.CLOUD_POST_ONLY):
                inputs_post = torch.stack([images[:, 1], images[:, 2], images[:, 3], images[:, 7]], dim=1)
                inputs_pre = torch.stack([images[:, 1+12], images[:, 2+12], images[:, 3+12], images[:, 7+12]], dim=1)
                
                # boolean pulse
                outputs = ((torch.sigmoid(model(inputs_post).squeeze(1)) > sub_cfg.TH_CLOUD) + (torch.sigmoid(model(inputs_pre).squeeze(1)) > sub_cfg.TH_CLOUD)).float()
            else:
                inputs = torch.stack([images[:, 1], images[:, 2], images[:, 3], images[:, 7]], dim=1)
                outputs = ((torch.sigmoid(model(inputs).squeeze(1)) > sub_cfg.TH_CLOUD)).float()
                
            clouds.append(outputs[0].cpu().numpy())
    
    # ensemble
    fire = np.stack(fires, axis=0).mean(axis=0) # C:ensemble, H, W -> H, W
    cloud = np.stack(clouds, axis=0).mean(axis=0) # C:ensemble, H, W -> H, W    
    
    # supression
    fire_bin = (fire > sub_cfg.TH_FIRE).astype(np.float32)
    cloud_bin = (cloud > 0.5).astype(np.float32)
    
    if sub_cfg.NO_CLOUD_MODEL:
        fire_bin_sup = fire_bin
    else:
        fire_bin_sup = fire_bin * (1 - cloud_bin)
    
    
    if phase == 'public':
        # metrics
        uuid_s.append(uuid)
        
        pos_ratio = 0
        pos_ratio_s.append(pos_ratio)
        
        false_positive = fire_bin_sup.sum()
        fp_s.append(false_positive)
            

    if sub_cfg.IS_IMG_SAVE:
        # optical image range selection
        image = (np.clip(images[0, 2:5].permute(1, 2, 0).cpu().numpy(), a_min=1, a_max=5000)/5000*255).astype(np.uint8)
        
        # save
        PATH_IMG = f'{save_preds_dir}{uuid}.png'
        path_imgs.append(PATH_IMG)
        cv2.imwrite(PATH_IMG, image)
        PATH_FIRE = f'{save_preds_dir}{uuid}_fire.png'
        cv2.imwrite(PATH_FIRE, (fire*255).astype(np.uint8))
        PATH_CLOUD = f'{save_preds_dir}{uuid}_cloud.png'
        cv2.imwrite(PATH_CLOUD, (cloud*255).astype(np.uint8))
        PATH_FIRE_BIN = f'{save_preds_dir}{uuid}_fire_bin.png'
        cv2.imwrite(PATH_FIRE_BIN, (fire_bin*255).astype(np.uint8))
        PATH_FIRE_SUP = f'{save_preds_dir}{uuid}_fire_sup.png'
        cv2.imwrite(PATH_FIRE_SUP, (fire_bin_sup*255).astype(np.uint8))
    
    if sub_cfg.debug and num_idx > sub_cfg.debug_sample:
        print(uuid)
        break
    
### CV SCORE ###
df = pd.DataFrame(
    { 'uuid': uuid_s, 'fp': fp_s, 'pos_ratio': pos_ratio_s }
)
fp_mean = df['fp'].mean()
print(f'FP: {fp_mean:.4f}')
NAME_CSV = f'036_{dt_now}_Sydony_f-th{sub_cfg.TH_FIRE:.2f}_c-th{sub_cfg.TH_CLOUD:.2f}_fp{fp_mean:.4f}_skip{sub_cfg.SKIP_THIS_FOLD}.csv'

if sub_cfg.IS_IMG_SAVE:
    df['PATH_IMG'] = path_imgs

df.to_csv(f'{sub_cfg.outdir}/metrics_{NAME_CSV}', index=False)

            
print(f'### Done -> {sub_cfg.outdir}')

