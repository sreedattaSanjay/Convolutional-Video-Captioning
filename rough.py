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
from prepro_vocab import build_vocab
import matplotlib.pyplot as plt
class PadSequence:
    '''
    This code is important and not mentioned anywhere about this
    It's a collate function to pad variable length sequences
    to be passed as arg to dataloader
    working but need to change whole code, so discarded
    
    '''
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        #print(batch)
        #for i in range(len(batch)):
            # print(batch[i]['video_ids'],len(batch[i]['c3d_feats']))

        sorted_batch = sorted(batch, key=lambda x: x['c3d_feats'].shape[0], reverse=True)
        #print(sorted_batch)
        # Get each sequence and pad it
        #unsorted=[x['c3d_feats'] for x in sorted_batch]
        sequences = [x['c3d_feats'] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
        
        # Don't forget to grab the labels of the *sorted* batch
        labels = list(map(lambda x: x['labels'], sorted_batch))
       
        video_ids=list(map(lambda x: x['video_ids'], sorted_batch))
        word_embed=map(lambda x: x['word_embed'], sorted_batch)
        gts=map(lambda x: x['gts'], sorted_batch)
        masks=map(lambda x: x['masks'], sorted_batch)
        #labels = torch.LongTensor(list(map(lambda x: x['labels'], sorted_batch)))
        return sequences_padded, lengths, labels,video_ids,list(word_embed),list(gts),list(masks)
        #,list(vid_embed),list(masks),list(gts)

def train(opt,EncoderRNN,DecoderCNN,Convcap,itow):
    '''
    training 
    initialize the models 
    pass the arg through the Convcap model
    output: /checkpoint/model.pth ----- trained model
    '''
    t_start = time.time()
    train_data=VideoDataset(opt, 'train')
    #####DataLODER#####
    collate_fn=PadSequence()
    train_loader=DataLoader(train_data, batch_size=opt["batch_size"],collate_fn=collate_fn,num_workers=opt['num_workers'], shuffle=True)
    print('[DEBUG] Loading train data ... %f secs' % (time.time() - t_start))
    #,word_embed,masks,gts
    for c3d_feat,lengths,labels,ids,word_embed,masks,gts in train_loader:
    #for data in train_loader: 
                print("came here")




def main(opt):
    videos = json.load(open('data/train_val_videodatainfo.json', 'r'))['sentences']
    test_videos=json.load(open('data/test_videodatainfo.json', 'r'))['sentences']

      
    video_caption = {}
    j=0
    for i in videos:
        j+=1
        #print(i)
        if i['video_id'] not in video_caption.keys():
            video_caption[i['video_id']] = {'captions': []}
        video_caption[i['video_id']]['captions'].append(i['caption'])
    for i in test_videos:
        j+=1
        #print(i)
        if i['video_id'] not in video_caption.keys():
            video_caption[i['video_id']] = {'captions': []}
        video_caption[i['video_id']]['captions'].append(i['caption'])
    
    vocab = build_vocab(video_caption)
    itow = {i + 2: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    wtoi['<eos>'] = 0
    itow[0] = '<eos>'
    wtoi['<sos>'] = 1
    itow[1] = '<sos>'
    

    train(opt,EncoderRNN,DecoderCNN,Convcap,itow)


if __name__ == '__main__':
    opt = opt.parse_opt()
    opt = vars(opt)
    #print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info1.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open('opt.json', 'w') as f:
        json.dump(opt, f)
    #print('save opt details to %s' % (opt_json))
    main(opt)