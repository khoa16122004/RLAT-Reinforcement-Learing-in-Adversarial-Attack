from torch.utils.data import Dataset
from torchvision import transforms, datasets
from typing import *
import os
from torchvision.transforms import ToTensor
from config import *
from torchvision.utils import save_image
from torchvision.io import read_image
from PIL import Image
from config import *

DATASETS = ["imagenet", "imagenet32", "cifar10", "mnist", "stl10", "restricted_imagenet"]

img_to_tensor = ToTensor()

def get_dataset(dataset: str, split=None) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""

    if dataset == "cifar10":
        return _cifar10(split)  

    if dataset == "cifar10_splits":
        return Cifar_10_splts(r"Splits")
    
    
def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "stl10":
        return 10
    elif dataset == "cifar10":
        return 10
    elif dataset == "cifar_splits":
        return 10
        




# class CustomImageDataset(Dataset):
#     def __init__(self, img_dir, transform=None, target_transform=None):
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label


def _cifar10(split: str) -> Dataset:
    dataset_path = os.path.join(os.getenv('PT_DATA_DIR', 'datasets'), 'dataset_cache')
    if split == "train":
        return datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform)
    elif split == "test":
        return datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform)

    else:
        raise Exception("Unknown split name.")


class Cifar_10_splts(Dataset):
    def __init__(self, img_dir, transform=transforms.Compose([transforms.ToTensor()]), target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.lines = []
        for label_folder in os.listdir(img_dir):
            label_path = os.path.join(img_dir, label_folder)
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                self.lines.append([file_path, int(label_folder)])
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_path, label = self.lines[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, idx
    
