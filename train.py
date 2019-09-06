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
    #collate_fn=PadSequence(),collate_fn=collate_fn,
    train_loader=DataLoader(train_data, batch_size=opt["batch_size"],num_workers=opt['num_workers'], shuffle=True)
    print('[DEBUG] Loading train data ... %f secs' % (time.time() - t_start))
    

    ##initialize encoder,decoder,model
    encoder=EncoderRNN.EncoderRNN(opt['dim_vid'],opt['dim_hidden'],bidirectional=opt['bidirectional'],rnn_cell=opt['rnn_type']).cuda()
    decoder=DecoderCNN.DecoderCNN(train_data.get_vocab_size()).cuda()
    convcap=Convcap.Convcap(encoder,decoder).cuda()


    ####initialize hyper params
    optimizer = optim.RMSprop(convcap.parameters(), lr=opt["learning_rate"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt["learning_rate_decay_every"],gamma=opt["learning_rate_decay_rate"])
    batchsize =opt['batch_size']
    cap_size= opt['max_len']
    nbatches = np.int_(np.floor((len(train_data)*1.)/batchsize))
    bestscore = .0
    batchsize_cap = batchsize*1
    max_tokens= opt['max_len']
    # print(batchsize,cap_size,batchsize_cap,max_tokens)
    # print("nbatches"+str(nbatches))

    itr = 0
    loss_graph=[]
    graph_x=[]
    for epoch in range(opt['epochs']):
        loss_train = 0.
        scheduler.step()
        
        for data in train_loader:
        #for c3d_feat,lengths,labels,word_embed,masks,gts in train_loader:
            #print("came here")
            print("iteration"+str(itr))
            itr+=1
            vid_feat=Variable(data['c3d_feats']).cuda()
            labels = Variable(data['labels'].type(torch.LongTensor)).cuda()
            mask = Variable(data['masks']).cpu()
            word_embed=Variable(data['word_embed']).cuda()
            cap=data['cap']
            # vid_feat=Variable(c3d_feat).cuda()
            # print(vid_feat.dtype)
            # print(vid_feat.shape)
            # labels = Variable(torch.FloatTensor(labels)).cuda()
            # mask = Variable(torch.FloatTensor(masks)).cpu()
            
            # word_embed= torch.stack([x for x in word_embed],dim=0)
            # print(word_embed.shape)
            
            # word_embed=Variable(word_embed).cuda()
            # lengths=lengths.type(torch.FloatTensor)
            # print(c3d_feat,word_embed)

            optimizer.zero_grad()
            wordact = convcap(vid_feat,labels,word_embed,'train')
            # print("//////////////////////////////////////////////")
            #print("1.wordact.shape"+str(wordact.shape))
            wordact = wordact[:,:,:-1]
            #print("2.wordact.shape"+str(wordact.shape))
            # print("////////////////////////////////////")
            labels = labels[:,1:]
            mask = mask[:,1:].contiguous()
            # print (wordact.shape)
            # print(batchsize_cap,max_tokens)
            wordact_t = wordact.permute(0, 2, 1).contiguous().view(\
            batchsize*(max_tokens-1), -1)
            #print(wordact_t.shape)
            wordclass_t = labels.contiguous().view(\
            batchsize*(max_tokens-1), 1)

            maskids = torch.nonzero(mask.view(-1)).numpy().reshape(-1)
            #print("mask Ids \t"+str(maskids))
            loss = F.cross_entropy(wordact_t[maskids, ...], \
            wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))
            ########### for visual##############################
            wordprobs = F.softmax(wordact_t).cpu().data.numpy()
            # print("word_class \t"+str(wordclass_t.shape)+"\t"+str(wordclass_t.dtype))
            # print(wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]))
            # print(wordact_t[maskids, ...])
            wordids = np.argmax(wordprobs, axis=1)
            for i in wordids:
                print(itow[i])
            # for i in wordclass_t[maskids, ...].contiguous().view(maskids.shape[0]):
            #     print(itow[i])
            print(cap)
             ############################ #####################  
            if itr%500 == 0:
                graph_x.append(itr)
                loss_graph.append(loss)

                

            loss_train = loss_train + loss.item()
            loss.backward()
            optimizer.step()
            print("loss"+str(loss_train))
            
        
        loss_train = (loss_train*1.)/(nbatches)
        print('[DEBUG] Training epoch %d has loss %f' % (epoch, loss_train))

        modelfn = osp.join(opt['checkpoint_path'], 'model_j_19_'+str(itr)+'.pth')
        torch.save({
                'epoch': epoch,
                'state_dict': convcap.state_dict(),              
                'optimizer' : optimizer.state_dict(),
                'loss':loss_train
            }, modelfn)
        print('time for epoch %f' % (time.time() - t_start))
    plt.plot(graph_x,loss_graph,'ro') 
    plt.show()

            




    

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