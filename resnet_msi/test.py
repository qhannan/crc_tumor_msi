import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import dataset
from dataset import med
#from model_vgg import VGG19
from model_resnet import Freeze_ResNet18
import os
import datetime
import sys
from datapre import get_files_withcsv_single
import pandas as pd
from sklearn.metrics import accuracy_score

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Input configuration
###########################################################################
#exts = ['.tif','.png']
#encoder = ['MSIMUT', 'MSS']
#filepath = '/media/dataa/MSS_MSI'
csvpath = 'target/crc_final.csv'
savepath_test = 'target/crctestlist.csv'
saveresults_test = 'target/crctestresults1.csv'
###########################################################################
get_files_withcsv_single(csvpath,savepath_test)

# Model save configuration
###########################################################################
modelname = 'RESNET18'
modelpath = './RESNET18-8.ckpt'
#logsave = 'target/crcdxRESNET18-6.txt'
###########################################################################


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###########################################################################
model = torch.load(modelpath,map_location=device)
###########################################################################


# Hyper-parameters
###########################################################################
batch_size = 128
###########################################################################


# Image preprocessing modules
transform = transforms.ToTensor()

files_pd_test = pd.read_csv(savepath_test)
files_source = pd.read_csv(csvpath)

test_dataset = med(files_pd_test,transforms = transform)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                        batch_size = batch_size,
                                        shuffle = False)
                                        #num_workers = 8,
                                        #pin_memory = True)


# Test the model

predicts = []
truelabels = []
imgpaths = []
outputs0 = []
outputs1 = []
outputs2 = []
outputs3 = []
print(len(test_loader))
with torch.no_grad():
    for i,(images, labels, imgpath) in enumerate(test_loader):
        print(i)
        images = images.to(device)
        labels = labels.to(device)
        model.eval()
        outputs = model(images)
        output_0 = outputs.data[:,0]
        outputs0 += output_0.tolist()
        output_1 = outputs.data[:,1]
        outputs1 += output_1.tolist()
#        output_2 = outputs.data[:,2]
#        outputs2 += output_2.tolist()
#        output_3 = outputs.data[:,3]
#        outputs3 += output_3.tolist()
        _,predict = torch.max(outputs.data,1)
        imgpaths += imgpath
        predicts += predict.tolist()
        truelabels += labels.tolist()
        #print(accuracy_score(labels.tolist(),predict.tolist()))
        #print(labels.tolist(),predict.tolist())
    files_source['truelabels'] = truelabels
    files_source['predicts'] = predicts
    files_source['imgpathforval'] = imgpaths
    files_source['output0'] = outputs0
    files_source['output1'] = outputs1
#    files_source['output2'] = outputs2
#    files_source['output3'] = outputs3
    files_source.to_csv(saveresults_test,index = False)
    acc = accuracy_score(truelabels,predicts)
    print(acc)
