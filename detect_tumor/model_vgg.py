from torchvision import models
from torch import nn

def VGG19(fc_out):
    model_ft = models.vgg19_bn(pretrained = False)
    num_features = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = nn.Linear(num_features,fc_out)
    return model_ft

#VGG19(9)