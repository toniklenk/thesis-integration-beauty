import os
import pandas as pd
import numpy as np

import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from scipy.stats import pearsonr, spearmanr
import torchvision.transforms.functional as TF
from scipy.io import loadmat


class ImageDatasetTaskonomy(object):
    """
    Iterate for Image dataset, ensures to always iterate in same order to
    yield images in same order as corresponding beauty ratings in file (which are according to the alphanmerical order of image names)
    """
    def __init__(self, img_dir, beauty_ratings_path=None):
        dir_img_list = list(f for f in os.listdir(img_dir) 
                            if os.path.isfile(os.path.join(img_dir, f)) and f.endswith('.jpg'))

        self.img_dir  = img_dir
        self.img_list =  sorted(dir_img_list)
        self.img_count = len(dir_img_list)
        if beauty_ratings_path is not None:
            self.beauty_ratings = pd.read_csv(beauty_ratings_path, header=None).mean(axis=1)

    def __iter__(self, transform = lambda x: x):
        """Initialize a iterator for one iteration over the dataset in alpha numerical order of image names"""
        self.img_pos = 0
        self.transform = transform #option to apply a transformation to all images, applies only for this iterator
        # ATTENTION: transform is commented out in iterator

        return self
    
    def __next__(self):
        if self.img_pos < self.img_count:
            img = torchvision.io.read_image(os.path.join(self.img_dir, self.img_list[self.img_pos]))
            self.img_pos += 1
            #return self.transform(img)
            return img
        else:
            self.img_pos = 0
            self.transform = lambda x: x
            raise StopIteration
        

class IntegrationCalculatorTaskonomy(object):
    def __init__(self, net, evaluation_layers_dict: dict):
        """
        Initialise with full PyTorch network model and layer names as in:
        _, eval_nodes = get_graph_node_names(net)
        
        ...and in the same structure as in the return_nodes dict in:
        https://pytorch.org/vision/stable/feature_extraction.html#api-reference
        
        """
        self.net = create_feature_extractor(net, return_nodes=evaluation_layers_dict)
        self.evalutation_layers = evaluation_layers_dict.keys()


    def integration_coeff(self, img):
        # TODO import image versions from matlab
        # TODO export activations to matlab for calculations
        img, img1, img2 = img.unsqueeze(0), img1.unsqueeze(0), img2.unsqueeze(0) 

        
        # activations for full image and image parts
        img_act, img1_act, img2_act = self.net(img), self.net(img1), self.net(img2)

        
        # average activation for image parts
        img12avg_act = { layer:None for layer in img_act.keys()}
        for layer in img_act.keys():
            img12avg_act[layer] = torch.stack((img1_act[layer], img2_act[layer]), dim=0).mean(dim=0)
        
        # calculate integration coefficient
        integration = { layer:None for layer in img_act.keys()}

        for (layer, act1, act2) in zip(integration.keys(), img_act.values(), img12avg_act.values()):

            integration[layer] = -pearsonr(act1.flatten(), act2.flatten())[0]

        return integration


def calculate_dataset_integration(dataset_iterator, integration_calculator: IntegrationCalculatorTaskonomy):
    """Calculate integration for whole dataset
        Input:
            dataset_iterator: class ImageDataset iterator that optionally specifies a image transformation
            integration: Integration calculator object that specifies DNN, layer to use and integration calculation procedure
        
        Output:
            DataFrame: images x layers
    """
    lst = []
    for img in dataset_iterator:
        lst.append(integration_calculator.integration_coeff(img))

    return pd.DataFrame(lst, columns=integration_calculator.evalutation_layers)

def correlate_integration_beauty(integration_ratings: pd.DataFrame, beauty_ratings: pd.DataFrame):
    return integration_ratings.aggregate(lambda x: spearmanr(x, beauty_ratings)[0], axis= 0)




