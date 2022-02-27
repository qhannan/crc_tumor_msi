from torchvision import models
from torch import nn

def Freeze_ResNet18(out_fc):
    model_ft = models.resnet18(pretrained = True)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features,out_fc)

    '''
    for idx, m in enumerate(model_ft.children()):
        if idx < 6:
            for param in m.parameters():
                param.requires_grad = False
        #print(idx, '->', m)
    '''
    return model_ft

#Freeze_ResNet18(2)