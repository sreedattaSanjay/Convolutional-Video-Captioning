import collections
import array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import csv
import pickle
import matplotlib.pyplot as plt
_nsre = re.compile('([0-9]+)') 
def SaveDictionary(dictionary,File):
    with open(File, "wb") as myFile:
        pickle.dump(dictionary, myFile)
        myFile.close()
def LoadDictionary(File):
    with open(File, "rb") as myFile:
        dict = pickle.load(myFile)
        myFile.close()
        return dict
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def read_binary_blob(filename):
    read_status = 1
    blob = collections.namedtuple('Blob', ['size', 'data'])
    f = open(filename, 'rb')
    s = array.array("i") # int32
    s.fromfile(f, 5)
    if len(s) == 5 :
        m = s[0]*s[1]*s[2]*s[3]*s[4]
        data_aux = array.array("f")
        data_aux.fromfile(f, m)
        data = np.array(data_aux.tolist())
        if len(data) != m:
            read_status = 0;
    else:
        read_status = 0;
    if not read_status:
        s = []
        blob_data = []
        b = blob(s, blob_data)
        return s, b, read_status
    blob_data = np.zeros((s[0], s[1], s[2], s[3], s[4]), np.float32)
    off = 0
    image_size = s[3]*s[4]
    for n in range(0, s[0]):
        for c in range(0, s[1]):
            for l in range(0, s[2]):
                # print n, c, l, off, off+image_size
                tmp = data[np.array(range(off, off+image_size))];
                blob_data[n][c][l][:][:] = tmp.reshape(s[3], -1);
                off = off+image_size;

    b = blob(s, blob_data)
    f.close()
    return s, b, read_status
mode="TRAIN_VAL"
path = '/home/sanjay/Documents/Video_convcap/data/'+str(mode)
action_folders = os.listdir(path)
action_folders.sort()
#### format video : array of c3d features
feat_dict={}
i=0
for a_folder in action_folders:
    print (a_folder)
    oox = os.listdir(path+'/'+a_folder)
    oox= sorted_aphanumeric(oox)
    #print (oox)
    feat_list=[]
    for ooi in oox:
        #print(ooi)
        s, b, read_status = read_binary_blob(path+'/'+a_folder+'/'+ooi)
        feat_list.append(b.data.squeeze())
        #print(len(feat_list))
    feat_array=np.asarray(feat_list)
    feat_dict[a_folder]=feat_array
    i+=1
    #if i==2:
       # break
# print (feat_dict)
# print (len(feat_dict))
SaveDictionary(feat_dict,str(mode).lower()+"_feat.pkl")

with open(str(mode).lower()+'_feat.csv', 'w') as f:
    w = csv.writer(f)
    for key,val in feat_dict.items():
        # print(key)
        # print(feat_dict[key].shape)
        w.writerow([key,val])

dic=LoadDictionary(str(mode).lower()+"_feat.pkl")
all_sizes=[]
all_vid=[]
    
for key in dic.keys():
    l=len(dic[key])
    # print (key)
    all_sizes.append(l)
    all_vid.append(key)
avg=sum(all_sizes)/len(all_sizes)
print(all_vid[1])
print(len(all_vid))
print(len([i for i in all_sizes if i > 10]) )
print(len([i for i in all_sizes if i < 10]) )
print(max(all_sizes),min(all_sizes),sum(all_sizes)/len(all_sizes))
# plt.plot(all_vid,all_sizes,'ro')
# plt.show()

