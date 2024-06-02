from architech.vgg import VGG
from architech.cifar_10 import ResNet26
import torch
from dataset import get_num_classes
import torch.nn as nn
def get_architecture(arch: str, dataset: str, pytorch_pretrained: bool=False) -> torch.nn.Module:
    num_classes = get_num_classes(dataset)
    if arch == "vgg13":
        return VGG('VGG13', num_classes)
    elif arch == "vgg16":
        return VGG('VGG16', num_classes)
    elif arch == "vgg19":
        return VGG('VGG19', num_classes)
    
    elif arch == "restnet26":
        print("using resnet")
        return ResNet26(in_channels=3, out_channels=num_classes)
    
class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear( in_features= input_size, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=output_size)
        )
    def forward(self, x):
        return self.classifier(x)
    
    



