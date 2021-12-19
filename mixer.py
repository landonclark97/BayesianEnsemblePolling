import numpy as np

import torch
from torchvision.models import resnet18
from torchvision import transforms
from torch import nn
from torch.optim import AdamW
from torch.autograd import Variable

smax = nn.Softmax(dim=0)
smax1 = nn.Softmax(dim=1)
tanner = nn.Tanh()

def mixer(posteriors):

    var = Variable(torch.tensor([[ 1.0856, -0.8727, -1.0467, -0.3036, -0.8310],
                    [ 1.1305, -0.9069, -1.0269, -0.4215, -0.8755],
                    [ 1.3684, -1.1563, -1.1994, -1.0811, -1.0957],
                    [ 1.3795, -1.0673, -1.1507, -1.0081, -0.9269],
                    [ 1.3253, -1.0454, -1.2259, -1.0163, -0.9788],
                    [-1.2328,  0.7034, -0.6763, -0.5674, -0.3547],
                    [-1.2114,  0.5667, -0.5478, -0.4705, -0.1469],
                    [-1.0714,  0.6762, -0.7921, -0.4475, -0.4275],
                    [-1.1060,  0.6635, -0.6612, -0.4947, -0.3856],
                    [-1.2217,  0.8230, -0.8849, -0.6001, -0.6727],
                    [-1.2825, -0.7937,  0.7530, -0.5329,  0.2618],
                    [-1.2594, -0.8180,  0.8236, -0.5790, -0.0265],
                    [-1.2669, -0.7824,  0.6863, -0.4080,  0.3088],
                    [-1.2590, -0.8892,  0.8456, -0.6614,  0.0965],
                    [-1.2986, -0.8917,  0.7082, -0.3872,  0.3150],
                    [-1.1374, -0.4058, -0.4792,  0.6497, -0.1319],
                    [-1.1294, -0.5416, -0.5795,  0.7231, -0.2406],
                    [-0.8556, -0.4658, -0.7128,  0.8271, -0.3468],
                    [-0.9014, -0.4902, -0.4249,  0.6939, -0.1466],
                    [-1.0471, -0.5649, -0.7457,  0.8953, -0.3982],
                    [-1.2787, -0.3477,  0.1949, -0.4789,  0.5846],
                    [-1.2768, -0.5939,  0.4543, -0.5011,  0.6238],
                    [-1.2214, -0.4529,  0.0990, -0.2054,  0.5920],
                    [-1.2613, -0.6009,  0.3667, -0.5372,  0.7000],
                    [-1.2831, -0.4998,  0.0436, -0.2978,  0.7352]]),requires_grad=True)

    return var

def mixer_forward(mix, post_pred, posteriors, b_size):

    y = tanner(torch.matmul(post_pred.reshape([16,25]),mix.to(torch.device("cuda"))))
    # c = post_pred.shape[1]

    # out = torch.empty((b_size,c))

    # for j in range(b_size):
        # for cl in range(c):
            # for inf in range(posteriors):
                # plc = post_pred[j,cl,inf] * y[j,inf]
                # out[j,cl] = out[j,cl] + plc

    return y
