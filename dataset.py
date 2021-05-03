import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision.io import read_image
import os
import numpy as np
import loss
from params import *

dirs = {'x':'x', 'z':'z'}
frameRange = 50

class ImgData(Dataset):
    def __init__(self, root_dir, label, transform = None):
        self.x_path = os.path.join(root_dir, dirs['x'])
        self.z_path = os.path.join(root_dir, dirs['z'])
        self.len = len(os.listdir(self.x_path)) // 2   #文件夹中包含两种图片；
        self.transform = transform
        self.label = label

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        z_path = os.path.join(self.z_path, f'{idx + 1:08d}_rgb.png')
        z_img = read_image(z_path)
        idxs = [x for x in range(max(idx - 20, 1), min(idx + 20, self.len)) if x != idx]
        x_idx = idxs[np.random.randint(0, len(idxs))]
        x_path = os.path.join(self.x_path, f'{x_idx:08d}_rgb.png')
        x_img = read_image(x_path)
        if self.transform:
            x_img = self.transform(x_img)
            z_img = self.transform(z_img)

        return {'x':x_img, 'z':z_img, 'label': self.label}


def to_float(img):
    img = img.to(dtype = torch.float32)
    return img

def get_train_dataset(root_dir, label):
    datasets = []
    label = loss.create_logloss_label(label_sz, rPos)
    for name in os.listdir(root_dir):
        if '.' not in name:
            img_dir = os.path.join(root_dir, name, 'siamFC')
            datasets.append(ImgData(img_dir, label, to_float))
            print(img_dir)
        #one video only
        #break
    train_dataset = ConcatDataset(datasets)
    return train_dataset

def get_special_dataset(root_dir, label, name):
    label = loss.create_logloss_label(label_sz, rPos)
    img_dir = os.path.join(root_dir, name, 'siamFC')
    return ImgData(img_dir, label, to_float)