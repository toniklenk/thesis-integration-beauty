import torch.utils.model_zoo # required to load nets
import torchsummary
from visualpriors.transforms import VisualPriorRepresentation, VisualPriorPredictedLabel
from torchvision.models.feature_extraction import get_graph_node_names


task = 'autoencoding'

VisualPriorRepresentation._load_unloaded_nets([task])
net = VisualPriorRepresentation.feature_task_to_net[task]




#torchsummary.summary(net, (3, 224, 224))

#get_graph_node_names(net)
 