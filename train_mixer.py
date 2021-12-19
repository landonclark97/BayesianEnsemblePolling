import numpy as np

import torch
from torchvision.models import resnet18
from torchvision import transforms
from torch import nn
from torch.optim import AdamW

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
print("using device:", device)

# try:

import mixer
import resnet_w_drop as res

import matplotlib.pyplot as plt

import glob
from PIL import Image

import random


BATCH_SIZE_PER_CLASS = 16
BATCH_PER_EPOCH = 8
EPOCHS = 40


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


def train(model, inf_engines):

    history = []

    mix_acc = []

    infers = len(inf_engines)

    m_i_acc = [[0],[0],[0],[0],[0]]
    m_i_clarity = [[0],[0],[0],[0],[0]]

    smax = nn.Softmax(dim=1)

    lr = 0.0005
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    for e in range(EPOCHS):

        correct = 0
        attempted = 0

        correct_i = [0]*infers
        attempted_i = [0]*infers

        opt = AdamW([model], lr=lr)

        print('beginning epoch:', e)

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

            loss = torch.autograd.Variable(torch.tensor([0]*BATCH_SIZE_PER_CLASS,dtype=torch.float32)).to(device)

            for c in range(classes):
                x = torch.empty((BATCH_SIZE_PER_CLASS,3,224,224)).float().to(device)
                y = torch.empty((BATCH_SIZE_PER_CLASS)).long().to(device)

                post_preds = torch.empty((BATCH_SIZE_PER_CLASS,classes,infers)).float().to(device)

                for j in range(BATCH_SIZE_PER_CLASS):
                    index = (b*BATCH_SIZE_PER_CLASS) + j
                    x[j,0:3,0:224,0:224] = data_list[c][index][0]
                    y[j] = data_list[c][index][1]


                res_i = [1]*infers
                with torch.no_grad():
                    for i, m in enumerate(inf_engines):
                        post_preds[:,:,i] = smax(m(x))
                        res_i[i] = torch.argmax(post_preds[:,:,i],dim=1)-y


                pred = mixer.mixer_forward(model, post_preds, infers, BATCH_SIZE_PER_CLASS)
                y_hat = smax(pred)

                res = torch.argmax(y_hat,dim=1)-y

                for j in range(BATCH_SIZE_PER_CLASS):
                    if res[j] == 0:
                        correct += 1
                    for inf_e in range(infers):
                        if res_i[inf_e][j] == 0:
                            correct_i[inf_e] += 1

                attempted += BATCH_SIZE_PER_CLASS

                for inf_e in range(infers):
                    attempted_i[inf_e] += BATCH_SIZE_PER_CLASS

                l = loss_fn(pred, y)
                loss += l



            mod1_clar = torch.empty((classes,classes))
            mod2_clar = torch.empty((classes,classes))
            mod3_clar = torch.empty((classes,classes))
            mod4_clar = torch.empty((classes,classes))
            mod5_clar = torch.empty((classes,classes))

            for i in range(classes):

                mod1_clar[i,:] = torch.div(model[i*classes,:],torch.linalg.vector_norm(model[i*classes,:]))
                mod2_clar[i,:] = torch.div(model[(i*classes)+1,:],torch.linalg.vector_norm(model[(i*classes)+1,:]))
                mod3_clar[i,:] = torch.div(model[(i*classes)+2,:],torch.linalg.vector_norm(model[(i*classes)+2,:]))
                mod4_clar[i,:] = torch.div(model[(i*classes)+3,:],torch.linalg.vector_norm(model[(i*classes)+3,:]))
                mod5_clar[i,:] = torch.div(model[(i*classes)+4,:],torch.linalg.vector_norm(model[(i*classes)+4,:]))


            clar1 = torch.abs(torch.linalg.det(mod1_clar))
            clar2 = torch.abs(torch.linalg.det(mod2_clar))
            clar3 = torch.abs(torch.linalg.det(mod3_clar))
            clar4 = torch.abs(torch.linalg.det(mod4_clar))
            clar5 = torch.abs(torch.linalg.det(mod5_clar))

            det_rate = 0.01

            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                        for p in model)

            loss = torch.mean(loss) - det_rate*(clar1+clar2+clar3+clar4+clar5) + l2_lambda*l2_norm
            # loss = torch.mean(loss)
            print(loss.tolist())
            history.append(loss.tolist())

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(model)

        ratio = float(correct)/float(attempted)
        print("accuracy of mixer:", ratio)
        mix_acc.append(ratio)

        ratio_i = [0.0]*infers

        for inf_e in range(infers):
            ratio_i[inf_e] = float(correct_i[inf_e])/float(attempted_i[inf_e])
            print("accuracy of model " + str(inf_e+1) + ":", ratio_i[inf_e])
            m_i_acc[inf_e].append(ratio_i[inf_e])

        mod1_clar = torch.empty((classes,classes))
        mod2_clar = torch.empty((classes,classes))
        mod3_clar = torch.empty((classes,classes))
        mod4_clar = torch.empty((classes,classes))
        mod5_clar = torch.empty((classes,classes))

        for i in range(infers):

            mod1_clar[i,:] = torch.div(model[i*classes,:],torch.linalg.vector_norm(model[i*classes,:]))
            mod2_clar[i,:] = torch.div(model[(i*classes)+1,:],torch.linalg.vector_norm(model[(i*classes)+1,:]))
            mod3_clar[i,:] = torch.div(model[(i*classes)+2,:],torch.linalg.vector_norm(model[(i*classes)+2,:]))
            mod4_clar[i,:] = torch.div(model[(i*classes)+3,:],torch.linalg.vector_norm(model[(i*classes)+3,:]))
            mod5_clar[i,:] = torch.div(model[(i*classes)+4,:],torch.linalg.vector_norm(model[(i*classes)+4,:]))

        clar = torch.linalg.det(mod1_clar).item()
        m_i_clarity[0].append(clar)
        print("model 1 clar: ", clar)
        clar = torch.linalg.det(mod2_clar).item()
        m_i_clarity[1].append(clar)
        print("model 2 clar: ", clar)
        clar = torch.linalg.det(mod3_clar).item()
        m_i_clarity[2].append(clar)
        print("model 3 clar: ", clar)
        clar = torch.linalg.det(mod4_clar).item()
        m_i_clarity[3].append(clar)
        print("model 4 clar: ", clar)
        clar = torch.linalg.det(mod5_clar).item()
        m_i_clarity[4].append(clar)
        print("model 5 clar: ", clar)


    m_i_acc[0] = m_i_acc[0][1:]
    m_i_acc[1] = m_i_acc[1][1:]
    m_i_acc[2] = m_i_acc[2][1:]
    m_i_acc[3] = m_i_acc[3][1:]
    m_i_acc[4] = m_i_acc[4][1:]


    m_i_clarity[0] = m_i_clarity[0][1:]
    m_i_clarity[1] = m_i_clarity[1][1:]
    m_i_clarity[2] = m_i_clarity[2][1:]
    m_i_clarity[3] = m_i_clarity[3][1:]
    m_i_clarity[4] = m_i_clarity[4][1:]

    return history, mix_acc, m_i_acc, m_i_clarity



res1 = res.load_resnet_w_dropout("./models/model1.pt", classes).to(device)
res2 = res.load_resnet_w_dropout("./models/model2.pt", classes).to(device)
res3 = res.load_resnet_w_dropout("./models/model3.pt", classes).to(device)
res4 = res.load_resnet_w_dropout("./models/model4.pt", classes).to(device)
res5 = res.load_resnet_w_dropout("./models/model5.pt", classes).to(device)

inf_engines = [res1, res2, res3, res4, res5]


model = mixer.mixer(len(inf_engines))


hist, mix_acc, m_i_acc, m_i_clarity = train(model, inf_engines)



# torch.save(model.state_dict(), "./models/mixer.pt")

t = range(len(mix_acc))


fig, axs = plt.subplots(3, 1)
fig.tight_layout()
axs[0].plot(range(len(hist)), hist, label="mixture loss")
axs[0].set(xlabel='Batchs', ylabel='Loss')
axs[1].plot(t, mix_acc, label="mixture acc")
axs[1].plot(t, m_i_acc[0], label="model 1 acc")
axs[1].plot(t, m_i_acc[1], label="model 2 acc")
axs[1].plot(t, m_i_acc[2], label="model 3 acc")
axs[1].plot(t, m_i_acc[3], label="model 4 acc")
axs[1].plot(t, m_i_acc[4], label="model 5 acc")
axs[1].set(xlabel='Epochs', ylabel='Accuracy')
axs[2].plot(t, m_i_clarity[0], label="model 1 clarity")
axs[2].plot(t, m_i_clarity[1], label="model 2 clarity")
axs[2].plot(t, m_i_clarity[2], label="model 3 clarity")
axs[2].plot(t, m_i_clarity[3], label="model 4 clarity")
axs[2].plot(t, m_i_clarity[4], label="model 5 clarity")
axs[2].set(xlabel='Epochs', ylabel='Clarity')

axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()


# t = range(len(hist))

# plt.plot(t,hist,label="mixer loss")

# plt.xlabel('Batch Number', fontsize=18)
# plt.ylabel('Cross-Entropy Loss', fontsize=18)
# plt.legend()
# plt.show()

# except Exception as e:

# print('failed with:', str(e))
# if str(device) == "cuda":
    # torch.cuda.ipc_collect()
    # torch.cuda.empty_cache()

res = torch.tensor([[ 1.0856, -0.8727, -1.0467, -0.3036, -0.8310],
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
                    [-1.2831, -0.4998,  0.0436, -0.2978,  0.7352]])


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
                        [-1.0944, -0.3789,  0.5103, -0.0899,  0.2582]])
