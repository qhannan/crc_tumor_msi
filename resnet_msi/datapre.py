import pandas as pd
from sklearn import preprocessing
import os
import random

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_exts(file,exts):
    return any(file.endswith(ext) for ext in exts)

def lbencoder(fite,label_s):
    le = preprocessing.LabelEncoder()
    le.fit(fite)
    label_n = le.transform(label_s)
    return label_n

def get_files_withcsv(path,savepath_train,savepath_val,val_ratio):
    zippedcsv = pd.read_csv(path)
    imgpath = zippedcsv['newpath']
    label = zippedcsv['subtype_label']
    zippedlist = list(zip(imgpath,label))
    random.shuffle(zippedlist)
    allimages = len(zippedlist)
    print(allimages)
    valset = zippedlist[:int(allimages*val_ratio)]
    valcsv = pd.DataFrame(valset,columns = ['imgpath','label_num'])
    valcsv.to_csv(savepath_val,index=False)
    trainset = zippedlist[int(allimages*val_ratio):]
    traincsv = pd.DataFrame(trainset,columns = ['imgpath','label_num'])
    traincsv.to_csv(savepath_train,index=False)

def get_files_withcsv_single(path,savepath_test):
    zippedcsv = pd.read_csv(path)
    imgpath = zippedcsv['imgpath']
    label = zippedcsv['msi_label']
    zippedlist = list(zip(imgpath,label))
    testset = zippedlist
    testcsv = pd.DataFrame(testset,columns = ['imgpath','label_num'])
    testcsv.to_csv(savepath_test,index=False)

def get_files(dir,exts,savepath_train,savepath_val,val_ratio,en):
    filelist = []
    labels = []
    for root,_,files in os.walk(dir):
        for file in files:
            if is_exts(file,exts):
                filelist.append(os.path.join(root,file))
                label = root.split('/')[-1]
                labels.append(label)
    labelnum = lbencoder(en,labels)
    zippedlist = list(zip(filelist,labels,labelnum))
    random.shuffle(zippedlist)
    #zippedlist = zippedlist[:1000]
    allimages = len(zippedlist)
    valset = zippedlist[:int(allimages*val_ratio)]
    valcsv = pd.DataFrame(valset,columns = ['imgpath','label','label_num'])
    valcsv.to_csv(savepath_val,index=False)
    trainset = zippedlist[int(allimages*val_ratio):]
    traincsv = pd.DataFrame(trainset,columns = ['imgpath','label','label_num'])
    traincsv.to_csv(savepath_train,index=False)

def get_files_test(dir,exts,savepath,en):
    filelist = []
    labels = []
    for root,_,files in os.walk(dir):
        for file in files:
            if is_exts(file,exts):
                filelist.append(os.path.join(root,file))
                label = root.split('/')[-1]
                labels.append(label)
    labelnum = lbencoder(en,labels)
    zippedlist = list(zip(filelist,labels,labelnum))
    zippedcsv = pd.DataFrame(zippedlist,columns = ['imgpath','label','label_num'])
    zippedcsv.to_csv(savepath,index=False)

def get_files_infer(dir,exts,max_size,savedir):
    mkdir(savedir)
    filelist = []
    for root,_,files in os.walk(dir):
        for file in files:
            if is_exts(file,exts):
                filelist.append(os.path.join(root,file))
    filecsv = pd.DataFrame(filelist,columns = ['imgpath'])
    for i in range(int(len(filecsv)/max_size)+1):
        filecsv.iloc[max_size*i:max_size*(i+1)].to_csv(os.path.join(savedir,'csv_{}.csv'.format(i)),index=False)

'''
exts = ['.png','.tif']
encoder = ['DEB', 'BACK', 'NORM', 'MUC', 'TUM', 'LYM', 'MUS', 'ADI', 'STR']
filepath = '/home/qc/Desktop/disk/CRC-VAL-HE-7K'
savepath = './files.csv'
get_files(filepath,exts,savepath,encoder)
'''
