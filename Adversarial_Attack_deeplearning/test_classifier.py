import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import os
import matplotlib.pyplot as plt
from arch import get_architecture
from dataset import get_dataset
from tqdm import tqdm
from ultils import *
import multiprocessing
from config import *

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # ------------------------------ Classifier -------------------------------
    clf = get_architecture(CLASSIFIER_ARCH, "cifar10").cuda()
    clf.load_state_dict(torch.load(CLASSIFIER_TRAINED))
    
    # ------------------------------ Dataloader -------------------------------
    test_dataset = get_dataset(DATASET, "test")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE,
                                 num_workers=2)

    acc_meter = AverageMeter()
    with torch.no_grad():
        for imgs, targets in tqdm(test_loader):
            imgs, targets = imgs.cuda(), targets.cuda()
            outputs = clf(imgs)
            acc = accuracy(outputs, targets)
            acc_meter.update(acc[0].item(), imgs.shape[0])
        
    print(f"Test Acc: {acc_meter.avg}\n")

        

    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()    
    