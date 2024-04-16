import os
import pandas as pd
import numpy as np

import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from scipy.stats import pearsonr, spearmanr
import torchvision.transforms.functional as TF
from scipy.io import loadmat


class ImageDataset(object):
    """
    Handles preparing images for input into activation extractors:
        
        - Load images (matlab arrays) from subfolder,
            in alphanumerical order (corresponding to beauty ratings in file).
        
        - Transform into PyTorch format
    
    This class provides a iterator to do so.
    """
    def __init__(self, img_dir, beauty_ratings_path=None):

        dir_img_list    = list(f for f in os.listdir(os.path.join(img_dir, 'full')))
        self.img_dir    = img_dir
        self.img_list   = sorted(dir_img_list)
        self.img_count  = len(dir_img_list)
        if beauty_ratings_path is not None:
            self.beauty_ratings = pd.read_csv(beauty_ratings_path, header=None).mean(axis=1)

    def __iter__(self, transform = lambda x: x):
        self.img_pos = 0
        return self
    
    def __next__(self):
        if self.img_pos < self.img_count:
            # load arrays (transformed in matlab)
            img_full = loadmat(os.path.join(self.img_dir,'full', self.img_list[self.img_pos]))["im"]
            img_v1 = loadmat(os.path.join(self.img_dir,'version1', self.img_list[self.img_pos]))["imv1"]
            img_v2 = loadmat(os.path.join(self.img_dir,'version2', self.img_list[self.img_pos]))["imv2"]
            
            # convert to input format of Taskonomy models
            img_full = torch.tensor(img_full).permute([2, 0, 1]).unsqueeze(0).float().flip(1)
            img_v1 = torch.tensor(img_v1).permute([2, 0, 1]).unsqueeze(0).float().flip(1)
            img_v2 = torch.tensor(img_v2).permute([2, 0, 1]).unsqueeze(0).float().flip(1)
            self.img_pos += 1
            return img_full, img_v1, img_v2
        else: # prepare for a possible next iteration
            self.img_pos = 0
            raise StopIteration


# --- Correlation (-Integration)
def correlation_coeff(net, img_full, img_v1, img_v2):
    """Calculate integration coefficient"""    
    # activations for full image and image parts
    with torch.no_grad():
        act_full, act_v1, act_v2 = net(img_full), net(img_v1), net(img_v2)

    integration = {}
    for (layer, act_full_, act_v1_, act_v2_) in zip(act_full.keys(), act_full.values(), act_v1.values(), act_v2.values()):
        # average activation for image parts
        act_avg_ = torch.stack((act_v1_, act_v2_), dim=0).mean(dim=0).flatten()
        act_full_ = act_full_.flatten()

        integration[layer] = pearsonr(act_full_, act_avg_)[0]

    return integration


def calculate_dataset_correlation(ImageDataset_iterator, net_tweaked):
    """Calculate integration for whole dataset"""
    lst = []
    for img_full, img_v1, img_v2 in ImageDataset_iterator:
        lst.append(correlation_coeff(net_tweaked, img_full, img_v1, img_v2))
    
    column_names = list(net_tweaked(torch.zeros(1,3,224,224).float().flip(1)).keys())
    return pd.DataFrame(lst, columns=column_names) 


def correlate_integration_beauty(correlation_ratings: pd.DataFrame, beauty_ratings: pd.DataFrame):
    return correlation_ratings.aggregate(lambda x: spearmanr(-x, beauty_ratings)[0], axis= 0)


# --- Self similarity
def self_similarity(net, img_v1, img_v2):
    """Calculate image self-similarity"""    
    # activations for image parts
    with torch.no_grad():
        act_v1, act_v2 = net(img_v1), net(img_v2)

    selfsimilarity = {}
    for (layer, act_v1_, act_v2_) in zip(act_v1.keys(), act_v1.values(), act_v2.values()):
        act_v1_ = act_v1_.flatten()
        act_v2_ = act_v2_.flatten()

        selfsimilarity[layer] = pearsonr(act_v1_, act_v2_)[0]

    return selfsimilarity


def calculate_dataset_self_similarity(ImageDataset_iterator, net_tweaked):
    """Calculate self-similarity for whole dataset"""
    lst = []
    for _, img_v1, img_v2 in ImageDataset_iterator:
        lst.append(self_similarity(net_tweaked, img_v1, img_v2))
    
    column_names = list(net_tweaked(torch.zeros(1,3,224,224).float().flip(1)).keys())
    return pd.DataFrame(lst, columns=column_names) 


# --- L2 norm
def l2_norm(net, img_full):
    """Calculate imgage L2-norm"""    
    # activation for full image
    with torch.no_grad():
        act_full = net(img_full)

    l2norm = {}
    for (layer, act_full_) in zip(act_full.keys(), act_full.values()):
        l2norm[layer] = act_full_.norm(p=2).item()

    return l2norm


def calculate_dataset_l2norm(ImageDataset_iterator, net_tweaked):
    """Calculate L2-norm for whole dataset"""
    lst = []
    for img_full, _, _ in ImageDataset_iterator:
        lst.append(l2_norm(net_tweaked, img_full))
    
    column_names = list(net_tweaked(torch.zeros(1,3,224,224).float().flip(1)).keys())
    return pd.DataFrame(lst, columns=column_names)


