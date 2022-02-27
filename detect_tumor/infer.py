import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import dataset
from dataset import med_infer
#from model_vgg import VGG
import os
import datetime
import sys
from datapre import get_files_infer
import pandas as pd
#from torchvision.utils import save_image
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#####################################################################
model_path = './VGG19_BN-61.ckpt'   #训练完成后最好的模型的位置
batch_size = 256
max_size = 50*batch_size

exts = ['.tif','.png']
filepath = '/media/data/CRC_DX'   #待infer文件位置
savedir = './csvfiless0.5'
saveresultsdir = './csvresutlss0.5' 
get_files_infer(filepath,exts,max_size,savedir)
print('listfiles done')

trans = transforms.ToTensor()
'''
####################################################
savedirlist = os.listdir(savedir)
newsavedir = []
for csvfile in savedirlist:
    if not 'result' in csvfile:
        if not '{}_results.csv'.format(csvfile[:-4]) in savedirlist:
            newsavedir.append(csvfile)
            print('add csvfile:{}'.format(csvfile))
        else:
            print(csvfile)
    else:
        print('results:{}'.format(csvfile))
###################################################
'''
with torch.no_grad():
    model = torch.load(model_path,map_location=device).eval()
    for csvfile in os.listdir(savedir):
        print(csvfile)
        csvpath = os.path.join(savedir,csvfile)
        files_pd = pd.read_csv(csvpath)
        custom_dataset = med_infer(files_pd,transforms = trans)
        test_loader = torch.utils.data.DataLoader(dataset = custom_dataset,
                                                batch_size = batch_size,
                                                shuffle = False)
                                                #num_workers = 8,
                                                #pin_memory = True)
        predict_labels = []
        imagenames = []
        for i, (images, imagesname) in enumerate(test_loader):
            print(i)
            images = images.to(device)
            outputs = model(images)
            _,predict = torch.max(outputs.data,1)
            predict_label = predict.tolist()
            #print(predict_label)
            predict_labels += predict_label
            imagenames += imagesname
        zipli = list(zip(imagenames,predict_labels))
        zipcsv = pd.DataFrame(zipli,columns = ['imgpath','label'])
        zipcsv.to_csv(os.path.join(saveresultsdir,'{}_results.csv'.format(csvfile[:-4])),index = False)
