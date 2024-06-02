# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shutil
import numpy as np
import torch
from PIL import Image


def save_concatenated_images(img1, img2, output_path):
    w, h = img1.size
    new_width = 2 * w
    new_height = h
    new_image = Image.new('RGB', (new_width, new_height))
    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (w, 0))
    new_image.save(output_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, targets):
    # Calculate the number of correct predictions
    correct = (outputs == targets).sum().item()
    # Total number of predictions
    total = targets.shape[0]
    # Calculate accuracy
    return correct / total


def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()


def measurement(n_measure, dim):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    aa = torch.normal(0, np.sqrt(1 / n_measure), size=(dim, n_measure)).cuda()
    return aa



def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()


# Set có huấn luyện trong quá trình lan truyền ko 
def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def copy_code(outdir):
    """Copies files to the outdir to store complete script with each experiment"""
    # embed()
    code = []
    exclude = set([])
    for root, _, files in os.walk("./code", topdown=True):
        for f in files:
            if not f.endswith('.py'):
                continue
            code += [(root,f)]

    for r, f in code:
        codedir = os.path.join(outdir,r)
        if not os.path.exists(codedir):
            os.mkdir(codedir)
        shutil.copy2(os.path.join(r,f), os.path.join(codedir,f))
    print("Code copied to '{}'".format(outdir))
