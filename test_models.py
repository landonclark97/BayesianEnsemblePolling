import numpy as np

import torch
from torchvision.models import resnet18
from torchvision import transforms
from torch import nn
from torch.optim import AdamW

torch.autograd.set_detect_anomaly(True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print("using device:", device)

# try:

import mixer
import resnet_w_drop as resn

import matplotlib.pyplot as plt

import glob
from PIL import Image

import random


BATCH_SIZE_PER_CLASS = 16
BATCH_PER_EPOCH = 8
EPOCHS = 100


classes = 5

root_dir = "./data/images/"

score_map = {
    'no_dr': 0,
    'mild': 1,
    'moderate': 2,
    'severe': 3,
    'proliferate': 4
}


no_dr = []
mild_dr = []
moderate_dr = []
severe_dr = []
proliferate_dr = []

data_list = [no_dr, mild_dr, moderate_dr, severe_dr, proliferate_dr]


trans = transforms.ToTensor()

print('loading images')
for i, k in enumerate(score_map.keys()):
    curr_dir = root_dir + k + '/'
    files = [f for f in glob.glob(curr_dir+"*.png")]
    for f in files:
        im = Image.open(f).convert('RGB')
        im_t = trans(im).float()
        i_tens = torch.tensor(i).long()
        # i_tens = nn.functional.one_hot(i_tens, num_classes=classes).float()
        d = [im_t, i_tens]
        data_list[i].append(d)

print('images per class:')
total_imgs = 0
for d in data_list:
    v = len(d)
    print(v)
    total_imgs += v

res = torch.tensor([[ 0.9659, -0.7911, -0.9025, -0.2986, -0.8327],
                    [ 1.0092, -0.8142, -0.8750, -0.4266, -0.8410],
                    [ 1.2326, -1.0014, -1.0378, -0.9649, -0.9391],
                    [ 1.2487, -0.9274, -0.9896, -0.9119, -0.7768],
                    [ 1.2004, -0.9147, -1.0642, -0.9373, -0.8288],
                    [-1.0747,  0.7529, -0.6991, -0.6035, -0.3834],
                    [-1.0535,  0.6043, -0.5731, -0.5064, -0.1610],
                    [-0.9252,  0.7265, -0.8168, -0.4781, -0.4598],
                    [-0.9532,  0.7116, -0.6898, -0.5350, -0.4108],
                    [-1.0696,  0.8820, -0.9005, -0.6420, -0.7243],
                    [-1.1228, -0.7584,  0.7663, -0.5564,  0.2205],
                    [-1.0991, -0.7496,  0.8458, -0.6061, -0.0820],
                    [-1.1079, -0.7719,  0.6858, -0.4313,  0.2862],
                    [-1.0981, -0.8448,  0.8612, -0.6900,  0.0520],
                    [-1.1390, -0.8972,  0.7186, -0.4057,  0.2848],
                    [-0.9983, -0.3947, -0.5062,  0.6544, -0.1551],
                    [-0.9913, -0.5299, -0.6225,  0.7320, -0.2644],
                    [-0.7285, -0.4559, -0.7423,  0.8446, -0.3821],
                    [-0.7733, -0.4862, -0.4534,  0.7045, -0.1721],
                    [-0.9144, -0.5613, -0.7888,  0.9165, -0.4315],
                    [-1.1184, -0.3298,  0.1754, -0.4990,  0.6056],
                    [-1.1168, -0.5843,  0.4446, -0.5300,  0.6342],
                    [-1.0622, -0.4421,  0.0768, -0.2149,  0.6058],
                    [-1.1001, -0.5845,  0.3511, -0.5651,  0.7345],
                    [-1.1225, -0.4885,  0.0185, -0.3120,  0.7596]]).to(device)

# res = torch.tensor([[ 1.0856, -0.8727, -1.0467, -0.3036, -0.8310],
#                     [ 1.1305, -0.9069, -1.0269, -0.4215, -0.8755],
#                     [ 1.3684, -1.1563, -1.1994, -1.0811, -1.0957],
#                     [ 1.3795, -1.0673, -1.1507, -1.0081, -0.9269],
#                     [ 1.3253, -1.0454, -1.2259, -1.0163, -0.9788],
#                     [-1.2328,  0.7034, -0.6763, -0.5674, -0.3547],
#                     [-1.2114,  0.5667, -0.5478, -0.4705, -0.1469],
#                     [-1.0714,  0.6762, -0.7921, -0.4475, -0.4275],
#                     [-1.1060,  0.6635, -0.6612, -0.4947, -0.3856],
#                     [-1.2217,  0.8230, -0.8849, -0.6001, -0.6727],
#                     [-1.2825, -0.7937,  0.7530, -0.5329,  0.2618],
#                     [-1.2594, -0.8180,  0.8236, -0.5790, -0.0265],
#                     [-1.2669, -0.7824,  0.6863, -0.4080,  0.3088],
#                     [-1.2590, -0.8892,  0.8456, -0.6614,  0.0965],
#                     [-1.2986, -0.8917,  0.7082, -0.3872,  0.3150],
#                     [-1.1374, -0.4058, -0.4792,  0.6497, -0.1319],
#                     [-1.1294, -0.5416, -0.5795,  0.7231, -0.2406],
#                     [-0.8556, -0.4658, -0.7128,  0.8271, -0.3468],
#                     [-0.9014, -0.4902, -0.4249,  0.6939, -0.1466],
#                     [-1.0471, -0.5649, -0.7457,  0.8953, -0.3982],
#                     [-1.2787, -0.3477,  0.1949, -0.4789,  0.5846],
#                     [-1.2768, -0.5939,  0.4543, -0.5011,  0.6238],
#                     [-1.2214, -0.4529,  0.0990, -0.2054,  0.5920],
#                     [-1.2613, -0.6009,  0.3667, -0.5372,  0.7000],
#                     [-1.2831, -0.4998,  0.0436, -0.2978,  0.7352]]).to(device)


res_reg = torch.tensor([[ 0.6666, -0.8020, -0.8395, -0.4842, -0.6341],
                        [ 0.6240, -0.8487, -0.7323, -0.5057, -0.6424],
                        [ 0.7470, -0.7715, -0.6250, -0.8733, -0.5091],
                        [ 0.6981, -0.8165, -0.7797, -0.5979, -0.5554],
                        [ 0.4699, -0.5422,  0.2839, -0.4625, -1.1993],
                        [-0.6386,  0.6672, -0.5263, -0.6540, -0.3546],
                        [-0.4584,  0.5255, -0.4763, -0.6878, -0.0508],
                        [-0.7796,  0.6865, -0.6564, -0.5343, -0.4110],
                        [-0.6874,  0.6941, -0.5968, -0.6907, -0.2947],
                        [-0.5771,  0.8436, -0.9730, -0.7122, -0.5578],
                        [-0.5100, -0.6054,  0.8000, -0.4742, -0.3035],
                        [-0.3989, -0.2637,  0.8189, -0.4451, -0.5401],
                        [-0.8549, -0.7193,  0.7559, -0.4411,  0.0112],
                        [-0.5004, -0.5393,  0.9381, -0.5260, -0.4127],
                        [ 0.0724, -0.7795,  0.3469, -0.5369,  0.6370],
                        [-0.6824, -0.3668, -0.3954,  0.5860, -0.1116],
                        [-0.9422, -0.3892, -0.4561,  0.6091, -0.2259],
                        [-0.4150, -0.4590, -0.5541,  0.7486, -0.3763],
                        [-0.7438, -0.4504, -0.4158,  0.6603, -0.2010],
                        [-0.2586, -0.5112, -0.7722,  0.7756, -0.2911],
                        [-0.2019, -0.3077, -0.1069, -0.4345,  0.7913],
                        [-0.3093, -0.6080,  0.1952, -0.5294,  0.8476],
                        [-0.2126, -0.3310, -0.2027, -0.2529,  0.9098],
                        [-0.3086, -0.4492,  0.0361, -0.5002,  0.8584],
                        [-1.0944, -0.3789,  0.5103, -0.0899,  0.2582]]).to(device)


def init_model(model, init, rate=0.25):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            init_model(module, init)
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
            setattr(model, name, new)
        if isinstance(module, nn.Conv2d):
            init(module.weight)


def enable_dropout(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            enable_dropout(module)
        if isinstance(module, nn.Dropout2d):
            module.train()


def test(inf_engines):

    mix_acc = []
    mix_reg_acc = []
    mix_reg_ub_acc = []
    mean_acc = []

    infers = len(inf_engines)

    m_i_acc = [[0],[0],[0],[0],[0]]

    smax = nn.Softmax(dim=1)

    samples = 5

    for e in range(EPOCHS):

        conf_mix = torch.empty((BATCH_SIZE_PER_CLASS,samples,classes)).to(device)
        conf_mix_reg = torch.empty((BATCH_SIZE_PER_CLASS,samples,classes)).to(device)
        conf_mean = torch.empty((BATCH_SIZE_PER_CLASS,samples,classes)).to(device)

        correct_res = 0
        correct_res_reg = 0
        correct_mean = 0
        attempted = 0

        correct_i = [0]*infers
        attempted_i = [0]*infers

        print('beginning run:', e)

        # ensure data is shuffled before training
        random.shuffle(no_dr)
        random.shuffle(no_dr)

        random.shuffle(mild_dr)
        random.shuffle(mild_dr)

        random.shuffle(moderate_dr)
        random.shuffle(moderate_dr)

        random.shuffle(severe_dr)
        random.shuffle(severe_dr)

        random.shuffle(proliferate_dr)
        random.shuffle(proliferate_dr)

        for b in range(BATCH_PER_EPOCH):

            for c in range(classes):
                x = torch.empty((BATCH_SIZE_PER_CLASS,3,224,224)).float().to(device)
                y = torch.empty((BATCH_SIZE_PER_CLASS)).long().to(device)

                post_preds = torch.empty((BATCH_SIZE_PER_CLASS,classes,infers)).float().to(device)

                for j in range(BATCH_SIZE_PER_CLASS):
                    index = (b*BATCH_SIZE_PER_CLASS) + j
                    x[j,0:3,0:224,0:224] = data_list[c][index][0]
                    y[j] = data_list[c][index][1]


                for s in range(samples):

                    with torch.no_grad():
                        for i, m in enumerate(inf_engines):
                            post_preds[:,:,i] = smax(m(x))


                    pred_res = smax(torch.matmul(post_preds.reshape([BATCH_SIZE_PER_CLASS,25]),res))
                    pred_res_reg = smax(torch.matmul(post_preds.reshape([BATCH_SIZE_PER_CLASS,25]),res_reg))
                    pred_mean = torch.mean(post_preds,dim=2)

                    conf_mix[:,s,:] = pred_res
                    conf_mix_reg[:,s,:] = pred_res_reg
                    conf_mean[:,s,:] = pred_mean


                res_i = [1]*infers
                for i, m in enumerate(inf_engines):
                    res_i[i] = torch.argmax(post_preds[:,:,i],dim=1)-y


                y_res = torch.argmax(torch.median(conf_mix,dim=1).values,dim=1)-y
                y_res_reg = torch.argmax(torch.median(conf_mix_reg,dim=1).values,dim=1)-y
                y_mean = torch.argmax(torch.median(conf_mean,dim=1).values,dim=1)-y


                for j in range(BATCH_SIZE_PER_CLASS):
                    if y_res[j] == 0:
                        correct_res += 1
                    if y_res_reg[j] == 0:
                        correct_res_reg += 1
                    if y_mean[j] == 0:
                        correct_mean += 1
                    for inf_e in range(infers):
                        if res_i[inf_e][j] == 0:
                            correct_i[inf_e] += 1

                attempted += BATCH_SIZE_PER_CLASS

                for inf_e in range(infers):
                    attempted_i[inf_e] += BATCH_SIZE_PER_CLASS


        ratio = float(correct_res)/float(attempted)
        print("accuracy of mixer:", ratio)
        mix_acc.append(ratio)

        ratio = float(correct_res_reg)/float(attempted)
        print("accuracy of regulated mixer:", ratio)
        mix_reg_acc.append(ratio)

        ratio = float(correct_mean)/float(attempted)
        print("accuracy of mean:", ratio)
        mean_acc.append(ratio)

        ratio_i = [0.0]*infers

        for inf_e in range(infers):
            ratio_i[inf_e] = float(correct_i[inf_e])/float(attempted_i[inf_e])
            print("accuracy of model " + str(inf_e+1) + ":", ratio_i[inf_e])
            m_i_acc[inf_e].append(ratio_i[inf_e])


    m_i_acc[0] = m_i_acc[0][1:]
    m_i_acc[1] = m_i_acc[1][1:]
    m_i_acc[2] = m_i_acc[2][1:]
    m_i_acc[3] = m_i_acc[3][1:]
    m_i_acc[4] = m_i_acc[4][1:]

    return mix_acc, mix_reg_acc, mean_acc, m_i_acc



res1 = resn.load_resnet_w_dropout("./models/model1.pt", classes).to(device)
res2 = resn.load_resnet_w_dropout("./models/model2.pt", classes).to(device)
res3 = resn.load_resnet_w_dropout("./models/model3.pt", classes).to(device)
res4 = resn.load_resnet_w_dropout("./models/model4.pt", classes).to(device)
res5 = resn.load_resnet_w_dropout("./models/model5.pt", classes).to(device)

inf_engines = [res1, res2, res3, res4, res5]


mix_acc, mix_reg_acc, mean_acc, m_i_acc = test(inf_engines)



t = range(len(mix_acc))


fig, axs = plt.subplots(3, 1)
fig.tight_layout()
axs[0].plot(t, mix_acc, label="mixture acc")
axs[0].plot(t, mix_reg_acc, label="reg mix acc")
axs[0].plot(t, mean_acc, label="mean acc")
axs[0].set(xlabel='Epochs', ylabel='Mixture Accuracy')
axs[1].plot(t, m_i_acc[0], label="model 1 acc")
axs[1].plot(t, m_i_acc[1], label="model 2 acc")
axs[1].plot(t, m_i_acc[2], label="model 3 acc")
axs[1].plot(t, m_i_acc[3], label="model 4 acc")
axs[1].plot(t, m_i_acc[4], label="model 5 acc")
axs[1].set(xlabel='Epochs', ylabel='Model Accuracy')
labels = ["mix acc","reg mix acc","mean acc","model 1 acc","model 2 acc","model 3 acc","model 4 acc","model 5 acc"]
results = [sum(mix_acc),sum(mix_reg_acc),sum(mean_acc),sum(m_i_acc[0]),sum(m_i_acc[1]),sum(m_i_acc[2]),sum(m_i_acc[3]),sum(m_i_acc[4])]
axs[2].bar(labels,results)
axs[0].legend()
axs[1].legend()
#axs[2].legend()
plt.show()
