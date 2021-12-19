import torch
from torchvision.models import resnet18
from torchvision import transforms
from torch import nn
from torch.optim import AdamW




def load_resnet_w_dropout(path, classes):
    def init_model(model, rate=0.25):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                init_model(module)
            if isinstance(module, nn.ReLU):
                new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
                setattr(model, name, new)
    def enable_dropout(model):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                enable_dropout(module)
            if isinstance(module, nn.Dropout2d):
                module.train()
    model = resnet18(num_classes=classes)
    init_model(model)
    model.load_state_dict(torch.load(path))
    model.train()
    # enable_dropout(model)
    return model
