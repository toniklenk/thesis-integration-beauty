import os
import pandas as pd

import torch
from scipy.stats import pearsonr
from scipy.io import loadmat

from collections import OrderedDict
from dataclasses import dataclass
from typing import Union # this is not needed for the code to work, just for the type-hint in the header
from __future__ import annotations # this is not needed for the code to work, just for a type-hint in a class methods

from taskonomy_network import TaskonomyDecoder, TaskonomyEncoder


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
            img_full = torch.tensor(img_full).permute([2, 0, 1]).unsqueeze(0)
            img_v1 = torch.tensor(img_v1).permute([2, 0, 1]).unsqueeze(0)
            img_v2 = torch.tensor(img_v2).permute([2, 0, 1]).unsqueeze(0)
            self.img_pos += 1
            return img_full, img_v1, img_v2
        else: # prepare for a possible next iteration
            self.img_pos = 0
            raise StopIteration


class Pattern_Generator(object):
    """
    Provides different subsets of nodes from activation pattern.

    - Same subsets of nodes across iteration for different images.
    - Takes care of different layers

    """
    def __init__(self, num_subsets: int, layer_shapes: OrderedDict, frac: float = .33) -> None:
        """
        Input:
            num_iterations: number of node subsets to be drawn (e.g. 10 000, 100 000)
            net: Taskonomy network where the activation patterns will be from
            frac: Fraction of nodes of a layer selected for each subset.
        """
        # random generator seed for each different subset
        self.num_subsets = num_subsets
        self.seeds = torch.randint(int(1E9), (num_subsets,))
        self.layer_shapes = layer_shapes # dict: keys: layer names; value: tensors with layer shape
        self.frac = frac

    def _generate_patterns(self, seed: int) -> OrderedDict:
        """
        Generates pattern for whole network
        with specified layer_shapes
        from the given seed
        """
        gen = torch.Generator().manual_seed(seed)
        return OrderedDict((name, torch.rand(shape, generator=gen) > (1-self.frac))
                           for name, shape in self.layer_shapes.items())
    
    def get_subset_pattern(self, subset_num):
        """Returns same subset of nodes each time it's called with the same subset num"""
        return self._generate_patterns(self.seeds[subset_num].item())


@dataclass(frozen=True)
class Activation_Pattern(object):
    """
    Handles activation pattern of a network to an image

    - Takes care of layers
    - Selects subset of nodes (takes pattern from class Pattern_Generator)
    - Calculates integration values for different layers
    """
    activation_pattern: dict
    # def __init__(self, activation_pattern: dict) -> None:
    #     """
    #     Initialize with activation pattern
    #     """
    #     self.activation_pattern = activation_pattern

    def __getitem__(self, layer_masks: dict):
        """
        Mask whole network with dict specifying tensor mask for each layer (as returned by class: Pattern_Generator)
        """
        return OrderedDict((layer_name,layer_activation[layer_mask]) 
                for (layer_name, layer_activation), layer_mask 
                in zip(self.activation_pattern.items(), layer_masks.values()))
    
    @staticmethod
    def average_with(first: Activation_Pattern, other: Activation_Pattern):
        """
        Calculate average of two activatin patterns
        """


def calculate_integration_coeff(pattern1: Activation_Pattern, pattern2: Activation_Pattern):
    """
    Calculate integration coeff from two acivation patterns.
    One is the activation to the full image and one the average of the two halfes.
    Returns a dict with the layer names (same as in the patterns) and
    the pearson correlation of each layer.
    """
    return OrderedDict((layer_name,-pearsonr(layer_pat2.flatten(), layer_pat1.flatten())[0])
            for (layer_name, layer_pat1), layer_pat2 in zip(pattern1.items(), pattern2.values()))
    

def taskonomy_net_layer_shapes(net: Union[TaskonomyEncoder, TaskonomyDecoder]) -> OrderedDict:
    """
    Creates a dictionary with the shapes of
    all the convolutional and fully connected layers of
    a Taskonomy network 
    """
    return OrderedDict((name,layer.data.shape)
            for name, layer in net.named_parameters()
            if "conv" in name or 'fc' in name)


def taskonomy_activation_layer_shapes(net_activation: OrderedDict) -> OrderedDict:
    """
    Creates a dictionary with the shapes of
    all the convolutional and fully connected layers of
    a Taskonomy network 
    """
    return OrderedDict((name, layer_activation.shape)
                       for name, layer_activation in net_activation.items())


def calculate_dataset_metrics(ImageDataset_iterator, net):
    """Calculate metrics for whole dataset"""

    def calculate_image_metrics(net, img_full, img_v1, img_v2):
        """Calculate correlation coefficients for all layers between full and average activation pattern"""
        """Calculate image self-similarity"""
        """Calculate imgage L2-norm"""

        # activations for full image and image parts
        with torch.no_grad():
            act_full, act_v1, act_v2 = net(img_full), net(img_v1), net(img_v2)

        correlations, selfsimilarity, l2norm = {}, {}, {}

        for (layer, act_full_, act_v1_, act_v2_) in zip(act_full.keys(), act_full.values(), act_v1.values(), act_v2.values()):
            # average activation for image parts
            act_avg_ = torch.stack((act_v1_, act_v2_), dim=0).mean(dim=0).flatten()
            
            l2norm[layer] = act_full_.norm(p=2).item()

            act_v1_ = act_v1_.flatten()
            act_v2_ = act_v2_.flatten()
            act_full_ = act_full_.flatten()

            correlations[layer] = pearsonr(act_full_, act_avg_)[0]

            selfsimilarity[layer] = pearsonr(act_v1_, act_v2_)[0]


        return correlations, selfsimilarity, l2norm


    lst_correlation, lst_selfsimilarity, lst_l2norm = [],[],[]

    for img_full, img_v1, img_v2 in ImageDataset_iterator:
        correlation, selfsimilarity, l2norm = calculate_image_metrics(net, img_full, img_v1, img_v2)

        lst_correlation.append(correlation)
        lst_selfsimilarity.append(selfsimilarity)
        lst_l2norm.append(l2norm)
    
    column_names = list(net(torch.zeros(1,3,256,256)).keys())
    return pd.DataFrame(lst_correlation, columns=column_names), pd.DataFrame(lst_selfsimilarity, columns=column_names), pd.DataFrame(lst_l2norm, columns=column_names)