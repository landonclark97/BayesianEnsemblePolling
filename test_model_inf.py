import numpy as np

import torch
from torchvision.models import resnet18
from torchvision import transforms
from torch import nn
from torch.optim import AdamW

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# try:

import mixer
import resnet_w_drop as resn

import matplotlib.pyplot as plt

import glob
from PIL import Image

import random


BATCH_SIZE_PER_CLASS = 40

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




fin = torch.tensor([[ 0.9659, -0.7911, -0.9025, -0.2986, -0.8327],
                    [-1.0747,  0.7529, -0.6991, -0.6035, -0.3834],
                    [-1.1228, -0.7584,  0.7663, -0.5564,  0.2205],
                    [-0.9983, -0.3947, -0.5062,  0.6544, -0.1551],
                    [-1.1184, -0.3298,  0.1754, -0.4990,  0.6056],
                    [ 1.0092, -0.8142, -0.8750, -0.4266, -0.8410],
                    [-1.0535,  0.6043, -0.5731, -0.5064, -0.1610],
                    [-1.0991, -0.7496,  0.8458, -0.6061, -0.0820],
                    [-0.9913, -0.5299, -0.6225,  0.7320, -0.2644],
                    [-1.1168, -0.5843,  0.4446, -0.5300,  0.6342],
                    [ 1.2326, -1.0014, -1.0378, -0.9649, -0.9391],
                    [-0.9252,  0.7265, -0.8168, -0.4781, -0.4598],
                    [-1.1079, -0.7719,  0.6858, -0.4313,  0.2862],
                    [-0.7285, -0.4559, -0.7423,  0.8446, -0.3821],
                    [-1.0622, -0.4421,  0.0768, -0.2149,  0.6058],
                    [ 1.2487, -0.9274, -0.9896, -0.9119, -0.7768],
                    [-0.9532,  0.7116, -0.6898, -0.5350, -0.4108],
                    [-1.0981, -0.8448,  0.8612, -0.6900,  0.0520],
                    [-0.7733, -0.4862, -0.4534,  0.7045, -0.1721],
                    [-1.1001, -0.5845,  0.3511, -0.5651,  0.7345],
                    [ 1.2004, -0.9147, -1.0642, -0.9373, -0.8288],
                    [-1.0696,  0.8820, -0.9005, -0.6420, -0.7243],
                    [-1.1390, -0.8972,  0.7186, -0.4057,  0.2848],
                    [-0.9144, -0.5613, -0.7888,  0.9165, -0.4315],
                    [-1.1225, -0.4885,  0.0185, -0.3120,  0.7596]]).to(device)





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


points = BATCH_SIZE_PER_CLASS


def test_inf(inf_engines):

    samples = 150

    sel = 0

    infers = len(inf_engines)

    conf_mix = torch.empty((samples,classes,points))
    conf_mix_reg = torch.empty((samples,classes,points))
    conf_mean = torch.empty((samples,classes,points))

    smax = nn.Softmax(dim=1)
    tanner = nn.Tanh()


    for c in range(classes):
        x = torch.empty((BATCH_SIZE_PER_CLASS,3,224,224)).float().to(device)
        y = torch.empty((BATCH_SIZE_PER_CLASS)).long().to(device)

        post_preds = torch.empty((BATCH_SIZE_PER_CLASS,classes,infers)).float().to(device)

        for j in range(BATCH_SIZE_PER_CLASS):
            x[j,0:3,0:224,0:224] = data_list[c][j+sel][0]
            y[j] = data_list[c][j+sel][1]

        for s in range(samples):

            with torch.no_grad():
                for i, m in enumerate(inf_engines):
                    post_preds[:,:,i] = smax(m(x))


            pred_res = smax(torch.matmul(post_preds.reshape([BATCH_SIZE_PER_CLASS,25]),res))
            pred_res_reg = smax(torch.matmul(post_preds.reshape([BATCH_SIZE_PER_CLASS,25]),res_reg))
            pred_mean = torch.mean(post_preds,dim=2)

            conf_mix[s,c,:] = pred_res[0:points,c]
            conf_mix_reg[s,c,:] = pred_res_reg[0:points,c]
            conf_mean[s,c,:] = pred_mean[0:points,c]


    return conf_mix, conf_mix_reg, conf_mean


res1 = resn.load_resnet_w_dropout("./models/model1.pt", classes).to(device)
res2 = resn.load_resnet_w_dropout("./models/model2.pt", classes).to(device)
res3 = resn.load_resnet_w_dropout("./models/model3.pt", classes).to(device)
res4 = resn.load_resnet_w_dropout("./models/model4.pt", classes).to(device)
res5 = resn.load_resnet_w_dropout("./models/model5.pt", classes).to(device)

inf_engines = [res1, res2, res3, res4, res5]


conf_mix, conf_mix_reg, conf_mean = test_inf(inf_engines)


mix_means = np.array([torch.mean(conf_mix[:,0,:],dim=0).numpy(),
                      torch.mean(conf_mix[:,1,:],dim=0).numpy(),
                      torch.mean(conf_mix[:,2,:],dim=0).numpy(),
                      torch.mean(conf_mix[:,3,:],dim=0).numpy(),
                      torch.mean(conf_mix[:,4,:],dim=0).numpy()])
mix_vars = np.array([torch.var(conf_mix[:,0,:],dim=0).numpy()*5,
                     torch.var(conf_mix[:,1,:],dim=0).numpy()*5,
                     torch.var(conf_mix[:,2,:],dim=0).numpy()*5,
                     torch.var(conf_mix[:,3,:],dim=0).numpy()*5,
                     torch.var(conf_mix[:,4,:],dim=0).numpy()*5])

mix_reg_means = np.array([torch.mean(conf_mix_reg[:,0,:],dim=0).numpy(),
                          torch.mean(conf_mix_reg[:,1,:],dim=0).numpy(),
                          torch.mean(conf_mix_reg[:,2,:],dim=0).numpy(),
                          torch.mean(conf_mix_reg[:,3,:],dim=0).numpy(),
                          torch.mean(conf_mix_reg[:,4,:],dim=0).numpy()])
mix_reg_vars = np.array([torch.var(conf_mix_reg[:,0,:],dim=0).numpy()*5,
                         torch.var(conf_mix_reg[:,1,:],dim=0).numpy()*5,
                         torch.var(conf_mix_reg[:,2,:],dim=0).numpy()*5,
                         torch.var(conf_mix_reg[:,3,:],dim=0).numpy()*5,
                         torch.var(conf_mix_reg[:,4,:],dim=0).numpy()*5])

mean_means = np.array([torch.mean(conf_mean[:,0,:],dim=0).numpy(),
                       torch.mean(conf_mean[:,1,:],dim=0).numpy(),
                       torch.mean(conf_mean[:,2,:],dim=0).numpy(),
                       torch.mean(conf_mean[:,3,:],dim=0).numpy(),
                       torch.mean(conf_mean[:,4,:],dim=0).numpy()])
mean_vars = np.array([torch.var(conf_mean[:,0,:],dim=0).numpy()*5,
                      torch.var(conf_mean[:,1,:],dim=0).numpy()*5,
                      torch.var(conf_mean[:,2,:],dim=0).numpy()*5,
                      torch.var(conf_mean[:,3,:],dim=0).numpy()*5,
                      torch.var(conf_mean[:,4,:],dim=0).numpy()*5])

c = range(1,classes+1)

fig, axs = plt.subplots(3, 1)
fig.tight_layout()
for p in range(points):
    axs[0].errorbar(c, mix_means[:,p], mix_vars[:,p], linestyle='None',marker='^')
axs[0].set(xlabel='Classes', ylabel='Polling Confidences')
axs[0].set_ylim([-0.1,1.1])
for p in range(points):
    axs[1].errorbar(c, mix_reg_means[:,p], mix_reg_vars[:,p], linestyle='None',marker='^')
axs[1].set(xlabel='Classes', ylabel='Regulated Polling Confidences')
axs[1].set_ylim([-0.1,1.1])
for p in range(points):
    axs[2].errorbar(c, mean_means[:,p], mean_vars[:,p], linestyle='None',marker='^')
axs[2].set(xlabel='Classes', ylabel='Mean Confidences')
axs[2].set_ylim([-0.1,1.1])
plt.show()
