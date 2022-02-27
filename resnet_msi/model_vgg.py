from torchvision import models
from torch import nn

def VGG19(fc_out):
    model_ft = models.vgg19_bn(pretrained = True)
    num_features = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = nn.Linear(num_features,fc_out)
    '''
    vggmodel = models.vgg19_bn(pretrained = True)
    vgg_s = vggmodel.state_dict()
    model_ft_s = model_ft.state_dict()
    vgg_s =  {k: v for k, v in vgg_s.items() if k in model_ft_s}
    model_ft_s.update(vgg_s)
    model_ft.load_state_dict(model_ft_s)
    '''
    return model_ft
#VGG(9)

'''
device = 'cuda'
model_ft = ResNet18().to(device)

params_to_update = model_ft.parameters()

feature_extract = False

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
'''
