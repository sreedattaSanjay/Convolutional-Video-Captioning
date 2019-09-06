import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import csv
import pickle
import opt
from torchnlp.word_to_vector import FastText
from torchnlp.word_to_vector import GloVe

def LoadDictionary(File):
    with open(File, "rb") as myFile:
        dict = pickle.load(myFile)
        myFile.close()
        return dict
def word_vec(list_idx):

    list_word_vec=[]
    vec=GloVe(name='840B',dim=300)
    for i,w in enumerate(list_idx):
        list_word_vec.append(vec[w])
    sentence_word_vec= torch.stack(list_word_vec)
    #print (sentence_word_vec.shape)
    return sentence_word_vec
class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length
    

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        f=open('data/caption.json','r')
        self.captions = json.load(f)
        info = json.load(open('data/info.json','r'))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        #print(self.ix_to_word["1"])
        
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))
        self.c3d_feats= LoadDictionary("train_val_feat.pkl")
       

        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load

        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['val'])
        
        # fc_feat = []
        # for dir in self.feats_dir:
        #     fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % (ix))))
        # fc_feat = np.concatenate(fc_feat, axis=1)
        # if self.with_c3d == 1:

        #     c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video%i.npy'%(ix)))
        #     c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
        #     fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
        label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)
        c3d_feat_dic=self.c3d_feats
        c3d_feat=c3d_feat_dic['video%i'%(ix)]

        captions = self.captions['video%i'%(ix)]['final_captions']
        
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]
                
        
        # random select a caption for this video
        cap_ix = random.randint(0, len(captions) - 1) 
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1
        final_cap=[]
        for idx in label:
            final_cap.append(self.ix_to_word[str(int(idx))])
        word_vec_array = word_vec(final_cap)
        #print(word_vec_array)
        data = {}
        data['c3d_feats'] = torch.from_numpy(c3d_feat).type(torch.FloatTensor)
        data['word_embed'] = word_vec_array.type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['gts'] = torch.from_numpy(gts).long()
        data['video_ids'] = 'video%i'%(ix)
        return data

    def __len__(self):
        return len(self.splits[self.mode])
opt=opt.parse_opt()
opt = vars(opt)
dataset = VideoDataset(opt, 'train')

