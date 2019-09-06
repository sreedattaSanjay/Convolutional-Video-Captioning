

import numpy as np
import csv
import time
import argparse
import opt
from dataloader import VideoDataset
import json
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import EncoderRNN,DecoderCNN,Convcap
train_data=VideoDataset(opt, 'train')
train_loader=DataLoader(train_data, batch_size=opt["batch_size"],num_workers=3, shuffle=False)
for data in train_loader:
           
            vid_feat=data['c3d_feats']
            labels = data['labels']
            mask = data['masks']
            word_embed=data['word_embed']
            vid_id=data['video_id']
