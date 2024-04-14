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
    Iterator for Image dataset.
    Ensures to iaterating images in same order as corresponding beauty ratings in file (alphanumerical order).
    """
    def __init__(self, img_dir, beauty_ratings_path=None):

        dir_img_list    = list(f for f in os.listdir(os.path.join(img_dir, 'full')))
        self.img_dir    = img_dir
        self.img_list   = sorted(dir_img_list)
        self.img_count  = len(dir_img_list)
        if beauty_ratings_path is not None:
            self.beauty_ratings = pd.read_csv(beauty_ratings_path, header=None).mean(axis=1)

    def __iter__(self, transform = lambda x: x):
        """Iterator for one iteration over the dataset in alpha numerical order of image names"""
        self.img_pos = 0
        return self
    
    def __next__(self):
        if self.img_pos < self.img_count:
            # load arrays transformed in matlab
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


def integration_coeff(net, img_full, img_v1, img_v2):
    """Calculate integration coefficient"""    
    # activations for full image and image parts
    with torch.no_grad():
        act_full, act_v1, act_v2 = net(img_full), net(img_v1), net(img_v2)

    integration = {}
    for (layer, act_full_, act_v1_, act_v2_) in zip(act_full.keys(), act_full.values(), act_v1.values(), act_v2.values()):
        # average activation for image parts
        act_avg_ = torch.stack((act_v1_, act_v2_), dim=0).mean(dim=0).flatten()
        act_full_ = act_full_.flatten()

        # calculate integration coefficient
        integration[layer] = -pearsonr(act_full_, act_avg_)[0]

    return integration


def calculate_dataset_integration(ImageDataset_iterator, net_tweaked):
    """Calculate integration for whole dataset"""
    lst = []
    for img_full, img_v1, img_v2 in ImageDataset_iterator:
        lst.append(integration_coeff(net_tweaked, img_full, img_v1, img_v2))
    
    column_names = list(net_tweaked(torch.zeros(1,3,224,224).float().flip(1)).keys())
    return pd.DataFrame(lst, columns=column_names) 


def correlate_integration_beauty(integration_ratings: pd.DataFrame, beauty_ratings: pd.DataFrame):
    return integration_ratings.aggregate(lambda x: spearmanr(x, beauty_ratings)[0], axis= 0)




