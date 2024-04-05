import os
from os.path import isfile, join

import pandas as pd
import numpy as np

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models.feature_extraction import create_feature_extractor
from scipy.stats import pearsonr
import torchvision.transforms.functional as TF


class ImageDataset(object):
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

        return self
    
    def __next__(self):
        if self.img_pos < self.img_count:
            img = Image.open(os.path.join(self.img_dir, self.img_list[self.img_pos]))
            self.img_pos += 1
            return self.transform(img)
        else:
            self.img_pos = 0
            self.transform = lambda x: x
            raise StopIteration


class IntegrationCalculator(object):
    def __init__(self, net, evaluation_layers_dict: dict):
        """
        Initialise with full PyTorch network model and layer names as in:
        _, eval_nodes = get_graph_node_names(net)
        
        ...and in the same structure as in the return_nodes dict in:
        https://pytorch.org/vision/stable/feature_extraction.html#api-reference
        
        """
        self.net = create_feature_extractor(net, return_nodes=evaluation_layers_dict)
        self.evalutation_layers = evaluation_layers_dict.keys()



    def __image_parts(self, img):
        pattern = self.__checkerboard(4)
        img = np.array(img)
        img_1, img_2 = np.where(pattern, img, 128), np.where(~pattern, img, 128)
        img_1, img_2 = Image.fromarray(img_1), Image.fromarray(img_2)
        return img_1, img_2 
    
    def __checkerboard(self, scale, output_size=640):
        board = np.indices((scale,scale)).sum(axis=0) % 2
        board = board.repeat(output_size/scale,axis=0).repeat(output_size/scale, axis=1)
        board = board[:,:, np.newaxis,].repeat(3,axis=2)
        return board.astype(dtype = np.bool_)


    def integration_coeff(self, img):
        # TODO: can probably remove Image format much earlier and do all operations with tensors
        img = img.resize((640,640))
        img1, img2 = self.__image_parts(img)
        img, img1, img2 = TF.to_tensor(img).unsqueeze(0), TF.to_tensor(img1).unsqueeze(0), TF.to_tensor(img2).unsqueeze(0)

        
        # activation for full image
        img_act = self.net(img)
        
        # average activation of image parts
        img1_act = self.net(img1)
        img2_act = self.net(img2)
        
        img12avg_act = { k:None for k in img_act.keys()}

        for layer in img_act.keys():
            img12avg_act[layer] = torch.stack((img1_act[layer], img2_act[layer]), dim=0).mean(dim=0)
        
        # calculate integration coefficient
        integration = { k:None for k in img_act.keys()}

        for (layer_name, a1, a2) in zip(integration.keys(), img_act.values(), img12avg_act.values()):
            integration[layer_name] = -pearsonr(a1.flatten(), a2.flatten())[0]

        return integration



#class MetricsCorrelationCalculator(object):
#    def __init(self):
#        pass

def calculate_dataset_integration(dataset_iterator, integration_calculator: IntegrationCalculator):
    """Calculate integration for whole dataset
        Input:
            dataset_iterator: class ImageDataset iterator that optionally specifies a image transformation
            integration: Integration calculator object that specifies DNN, layer to use and integration calculation procedure
    """
    lst = []
    for img in dataset_iterator:
        lst.append(integration_calculator.integration_coeff(img))

    return pd.DataFrame(lst, columns=integration_calculator.evalutation_layers)

def correlate_integration_beauty(integration_ratings: pd.DataFrame, beauty_ratings: pd.DataFrame):
    return integration_ratings.aggregate(lambda x: pearsonr(x, beauty_ratings)[0], axis= 0)


class GLM_Calculator(object):
    def __init__(self):
        """Configure GLM"""
        pass

    