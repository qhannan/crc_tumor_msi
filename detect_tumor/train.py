import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset import *
from model_vgg import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import datetime
import sys
from datapre import get_files
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Input configuration
###########################################################################
exts = ['.tif','.png']
encoder = ['DEB', 'BACK', 'NORM', 'MUC', 'TUM', 'LYM', 'MUS', 'ADI', 'STR'] 
filepath = '/media/datab/PlosMedicine_data/CRC-VAL-HE-7K/' # 训练数据文件夹
savepath_train = './trainlist.csv'
savepath_val = './vallist.csv'
valratio = 0.2  # train / validatation 比例
###########################################################################
get_files(filepath,exts,savepath_train,savepath_val,valratio,encoder)


# Model save configuration
###########################################################################
modelname = 'VGG19_BN'
###########################################################################
log_path = '{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),modelname)
mode_path = os.path.join('./checkpoints','train',log_path)
mkdir(mode_path)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG19(9).to(device)  # n分类，VGG19(n)

# Hyper-parameters
###########################################################################
epoch_ratio = 3
min_epochs = 3
max_epochs = 100
learning_rate = 0.0003
batch_size = 32
val_batch_size = 256
weight_decay = 0.0001
###########################################################################
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))


# Image preprocessing modules
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

files_pd_train = pd.read_csv(savepath_train)
files_pd_val = pd.read_csv(savepath_val)

train_dataset = med(files_pd_train,transforms = transform)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8,
                                            pin_memory = True)
val_dataset = med(files_pd_val,transforms = transform)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                        batch_size = val_batch_size,
                                        shuffle = False,
                                        num_workers = 8,
                                        pin_memory = True)


# Train the model
max_acc = 0
best_epoch = 0
num_epochs = min_epochs
total_step = len(train_loader)
print(total_step)
try:
    for epoch in range(max_epochs):
        if epoch < num_epochs:
            # Training
            print("Current Learning rate：%f" % (optimizer.param_groups[0]['lr']))
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                #outputs = torch.sigmoid(outputs)
                #print(outputs.data)
                #print(labels.data)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if (i+1) % 10 == 0:
                    with open(os.path.join('checkpoints','train',log_path,'Log.txt'), 'a') as log:
                        log.write("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}\n"
                                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                if (i+1) % 3 ==0:
                    print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

            torch.save(model, os.path.join(mode_path,'{}-{}.ckpt'.format(modelname,epoch)))

            # Validation
            predicts = []
            truelabels = []
            with torch.no_grad():
                for i,(images, labels) in enumerate(val_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    val_model = model.eval()
                    outputs = model(images)
                    _,predict = torch.max(outputs.data,1)
                    predicts += predict.tolist()
                    truelabels += labels.tolist()
                acc = accuracy_score(truelabels,predicts)
                print(acc)
                if acc > max_acc:
                    max_acc = acc
                    best_epoch = epoch
                    num_epochs = max(min_epochs,int(epoch * epoch_ratio))
                    print('Find the best epoch:{}, the acc of validation set is {}'.format(epoch,acc))
        else:
            break
    print('Training Done, the best epoch is {}, the best acc is {}'.format(best_epoch,max_acc))
    with open(os.path.join('checkpoints','train',log_path,'Log.txt'),'a') as log:
        log.write('Training Done, best epoch:{}, best acc:{}'.format(best_epoch,max_acc))



except KeyboardInterrupt:
    torch.save(model, os.path.join(mode_path,'{}-interupt.ckpt'.format(modelname)))
    print('saved interupt')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
'''
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
'''
