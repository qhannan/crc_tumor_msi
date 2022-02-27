import pandas as pd
import shutil
from sklearn import preprocessing
'''
trainfiles = pd.read_csv('./finaltrain.csv')
imgpath = trainfiles['imgpath']
newimgpath = []

def trans(x):
    return '/media/dataa/CRC/MSI_MSS' + x[32:]

newimgpath = trainfiles['imgpath'].apply(trans)

zippedlist = list(zip(imgpath,newimgpath))
for i in zippedlist:
    sourcepath = i[0]
    targetpath = i[1]
    shutil.copy(sourcepath,targetpath)
trainfiles['newpath'] = newimgpath
trainfiles.to_csv('testtrain.csv',index = False)
'''
def lbencoder(fite,label_s):
    le = preprocessing.LabelEncoder()
    le.fit(fite)
    label_n = le.transform(label_s)
    return label_n
csv = pd.read_csv('testtrain.csv')
label = csv['msi_label']
fite = [0,1]
label_n = lbencoder(fite,label)
csv['msilabel'] = label_n
csv.to_csv('crctrain.csv',index = False)