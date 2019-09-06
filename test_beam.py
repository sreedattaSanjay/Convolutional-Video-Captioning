import sys
sys.path.insert(0, '/home/sanjay/Documents/Video_convcap/model')
import os
import re
import os.path as osp
import argparse
import numpy as np 
import json
import time
import opt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm 
from model import EncoderRNN,DecoderCNN,Convcap
from dataloader import VideoDataset
from beamsearch import beamsearch 
from evaluate import language_eval

def build_vocab(vids):
    count_thr = 1
    # count up the number of words
    counts = {}
    for vid, caps in vids.items():
        for cap in caps['captions']:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            for w in ws:
                counts[w] = counts.get(w, 0) + 1
    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')
    for vid, caps in vids.items():
        
        caps = caps['captions']
        vids[vid]['final_captions'] = []
        for cap in caps:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            caption = [
                '<sos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
            vids[vid]['final_captions'].append(caption)
    
    return vocab



def test(opt,EncoderRNN,DecoderCNN,Convcap,itow,wtoi,modelfn=None):
    t_start = time.time()
    t_start = time.time()
    
    test_data=VideoDataset(opt, 'train')
    test_loader=DataLoader(test_data, batch_size=opt["batch_size"],num_workers=3, shuffle=False)
    print('[DEBUG] Loading test data ... %f secs' % (time.time() - t_start))

    batchsize =opt['batch_size']
    cap_size= opt['max_len']
    nbatches = np.int_(np.floor((len(test_data)*1.)/batchsize))
    bestscore = .0
    batchsize_cap = batchsize*1
    max_tokens= opt['max_len']

    if(modelfn is not None):
        encoder=EncoderRNN.EncoderRNN(opt['dim_vid'],opt['dim_hidden'],bidirectional=opt['bidirectional'],rnn_cell=opt['rnn_type']).cuda()
        decoder=DecoderCNN.DecoderCNN(test_data.get_vocab_size()).cuda()
        convcap=Convcap.Convcap(encoder,decoder).cuda()
        print('[DEBUG] Loading checkpoint %s' % modelfn)
        checkpoint = torch.load(modelfn)
        print(checkpoint.keys())
        print(checkpoint['epoch'])
        print(checkpoint['state_dict'])
        convcap.load_state_dict(checkpoint['state_dict'])
        
    convcap.train(False)
    pred_captions = []
    itr=0
    for data in test_loader:
        print("iteration"+str(itr))
        itr+=1
        vid_feat=Variable(data['c3d_feats']).cuda()
        labels = Variable(data['labels']).cuda()
        mask = Variable(data['masks']).cpu()
        word_embed=Variable(data['word_embed']).cuda()
        vid_id=data['video_ids']
        print(vid_id)
        wordclass_feed = np.zeros((batchsize, max_tokens), dtype='int64')
        wordclass_feed[:,0] = 1 #
        outcaps = np.empty((batchsize, 0)).tolist()
        for j in range(max_tokens-1):
            wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()
            wordact = convcap(vid_feat,wordclass,word_embed,'test')
            wordact = wordact[:,:,:-1]
            wordact_t = wordact.permute(0, 2, 1).contiguous().view(batchsize*(max_tokens-1), -1)
            print("convcap output"+str(wordact_t.shape))

            wordprobs = F.softmax(wordact_t).cpu().data.numpy()
            
            wordids = np.argmax(wordprobs, axis=1)
            probs=np.max(wordprobs,axis=1)
            
            for k in range(batchsize):
                word = itow[wordids[j+k*(max_tokens-1)]]
                outcaps[k].append(word)
                if(j < max_tokens-1):
                    wordclass_feed[k, j+1] = wordids[j+k*(max_tokens-1)]
            for j in range(batchsize):
                num_words = len(outcaps[j]) 
                if 'eos' in outcaps[j]:
                    num_words = outcaps[j].index('eos')
                outcap = ' '.join(outcaps[j][:num_words])
                pred_captions.append({'vid_id': vid_id, 'caption': outcap})
            print(outcap)
            print(probs)
        if(itr==20):
            break
    scores = language_eval(pred_captions, '/home/sanjay/Documents/Video_convcap/output', 'test')

    return scores
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
    test(opt,EncoderRNN,DecoderCNN,Convcap,itow,wtoi,modelfn='/home/sanjay/Documents/Video_convcap/checkpoint/model36513.pth')        

if __name__ == '__main__':
    opt = opt.parse_opt()
    opt = vars(opt)
    #print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    main(opt)