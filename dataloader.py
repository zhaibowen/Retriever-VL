import os
import cv2
import time
import copy
import json
import torch
import inspect
import random
import pickle
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from config import RetrieverVLConfig_medium, RetrieverVLConfig_medium_finetune
from torchvision.transforms.functional import rotate, resize, adjust_brightness, adjust_saturation, adjust_hue, adjust_contrast, InterpolationMode

def make_label(x, IGNORE_INDEX):
    label = list(map(lambda x: x if x == 1 else IGNORE_INDEX, x))
    index = np.where(np.array(x) == 1)[0]
    for i in range(0, len(index), 2):
        i0 = index[i] + 4
        i1 = index[i+1] if i+1 < len(index) else len(x)
        label[i0 : i1] = x[i0 : i1]
    return label

class GPTDataset(Dataset):
    def __init__(self, token_dump_path, text_sequence_length, transform=None):
        self.transform = transform
        self.images, self.tokens = [], []

        for tp in token_dump_path:
            images, tokens = pickle.load(open(tp, 'rb'))
            self.images.extend(images)
            self.tokens.extend(tokens)

        # self.images = self.images[:100000]
        # self.tokens = self.tokens[:100000]
        
        self.tokens = list(map(lambda x: x[:text_sequence_length], self.tokens)) # truncate
        self.tokens = list(map(lambda x: x + [0] * (text_sequence_length - len(x)), self.tokens)) # padding
        
        IGNORE_INDEX = -100
        self.labels = list(map(lambda x: make_label(x, IGNORE_INDEX), self.tokens))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        if self.transform:
            img = self.transform(img)
        return img / 255., self.tokens[index], self.labels[index]

def image_loader(img_path, img_size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    
    _, H, W = img.shape
    scale_ratio = img_size / max(H, W)
    scale_h = int(H * scale_ratio)
    scale_w = int(W * scale_ratio)

    img = resize(img, [scale_h, scale_w], InterpolationMode.NEAREST)
    img_new = torch.zeros((3, img_size, img_size), dtype=img.dtype)
    img_new[:, :scale_h, :scale_w] = img

    return (img_new / 255.).unsqueeze(0)

class ImageTransformer(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def get_size(self, scale_h):
        t_top = s_top = 0
        t_bottom = s_bottom = self.img_size
        if scale_h > self.img_size:
            t_top = 0
            t_bottom = self.img_size
            s_top = np.random.randint(0, scale_h - self.img_size)
            s_bottom = s_top + self.img_size
        elif scale_h < self.img_size:
            t_top = np.random.randint(0, self.img_size - scale_h)
            t_bottom = t_top + scale_h
            s_top = 0
            s_bottom = scale_h
        return t_top, t_bottom, s_top, s_bottom

    def __call__(self, img):
        _, H, W = img.shape
        # cv2.imshow('x', img.permute(1,2,0).numpy())
        scale_ratio = np.random.uniform(0.8, 1.2) * self.img_size / max(H, W)
        scale_h = np.random.uniform(0.9, 1.1) * H
        scale_w = np.random.uniform(0.9, 1.1) * W
        scale_h2 = int(scale_h * scale_ratio)
        scale_w2 = int(scale_w * scale_ratio)

        angle = np.random.uniform(-5.0, 5.0)
        brightness = np.random.uniform(0.9, 1.1)
        contrast = np.random.uniform(0.9, 1.1)
        saturation = np.random.uniform(0.9, 1.1)
        hue = np.random.uniform(-0.02, 0.02)
        
        img = resize(img, [scale_h2, scale_w2], InterpolationMode.NEAREST)
        img_new = torch.zeros((3, self.img_size, self.img_size), dtype=img.dtype)
        t_top, t_bottom, s_top, s_bottom = self.get_size(scale_h2)
        t_left, t_right, s_left, s_right = self.get_size(scale_w2)
        img_new[:, t_top : t_bottom, t_left : t_right] = img[:, s_top : s_bottom, s_left : s_right]
        
        img_new = rotate(img_new, angle)
        img_new = adjust_brightness(img_new, brightness)
        img_new = adjust_contrast(img_new, contrast)
        img_new = adjust_saturation(img_new, saturation)
        img_new = adjust_hue(img_new, hue)
        # cv2.imshow('y', img_new.permute(1,2,0).numpy())
        # cv2.waitKey(0)
        return img_new

class RandSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.batch_size = batch_size
        self.order = list(range(len(data_source)))
        self.total_size = len(self.order) - len(self.order) % self.batch_size
        if not drop_last: self.total_size += batch_size

        if shuffle: random.shuffle(self.order)
        self.groups = []
        for i in range(0, self.total_size, self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def shuffle(self, epoch=0):
        random.shuffle(self.order)
        self.groups = []
        for i in range(0, self.total_size, self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def __iter__(self):
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)

class DistRandSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()
        self.batch_size = batch_size
        self.order = list(range(len(data_source)))
        self.total_size = len(self.order) - len(self.order) % (self.num_replicas * self.batch_size)
        if not drop_last: self.total_size += self.num_replicas * self.batch_size

        if shuffle:
            g = torch.Generator()
            g.manual_seed(-1)
            self.order = torch.randperm(len(self.order), generator=g).tolist()
        self.groups = []
        for i in range(self.rank * self.batch_size, self.total_size, self.num_replicas * self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def shuffle(self, epoch):
        g = torch.Generator()
        g.manual_seed(epoch)
        self.order = torch.randperm(len(self.order), generator=g).tolist()
        self.groups = []
        for i in range(self.rank * self.batch_size, self.total_size, self.num_replicas * self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def __iter__(self):
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)

def FixCollector(batch):
    images, tokens, labels = zip(*batch)
    images = torch.stack(images)
    tokens = torch.from_numpy(np.array(tokens)).to(torch.long)
    labels = torch.from_numpy(np.array(labels)).to(torch.long)
    return images, tokens, labels

if __name__ == "__main__":
    cur_dir = "/home/work/disk/vision/retriever-vl"
    token_dump_path = [
        # "checkpoint/tokens_coco.pkl", # 592000
        # "checkpoint/tokens_LLAVA.pkl" # 553000
        "checkpoint/tokens_Instruct_LLAVA.pkl"
    ]
    config = RetrieverVLConfig_medium()
    finetune_config = RetrieverVLConfig_medium_finetune()
    revised_params = list(filter(lambda x: x[0][0] != '_', inspect.getmembers(finetune_config)))
    for rp, value in revised_params:
        setattr(config, rp, value)
    token_dump_path = list(map(lambda x: os.path.join(cur_dir, x), token_dump_path))

    dataset = GPTDataset(token_dump_path, config.text_sequence_length, transform=ImageTransformer(config.img_size))
    sampler = RandSampler(dataset, batch_size=4, drop_last=True, shuffle=True)
    dataloader = DataLoader(dataset, num_workers=0, pin_memory=True, collate_fn=FixCollector, batch_sampler=sampler)
    
    for j, data in enumerate(dataloader):
        a = 1