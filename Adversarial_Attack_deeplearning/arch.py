from architech.vgg import VGG
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


class CNN(nn.Module):
    def __init__(self, input_size: int, output_size: int): # input_size = 3*6*6
        super(CNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=output_size)
        )
    def forward(self, x):
        return self.classifier(x)