import numpy as np

import torch
from torchvision.models import resnet18
from torchvision import transforms
from torch import nn
from torch.optim import AdamW

torch.autograd.set_detect_anomaly(True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("using device:", device)

try:

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


    def train(model):

        history = []

        model.train()

        lr = 0.001
        loss_fn = nn.CrossEntropyLoss(reduction='mean')

        for e in range(EPOCHS):

            if (e+1) % 5 == 0:
                lr = lr*0.9

            opt = AdamW(model.parameters(), lr=lr)

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
                    # y = torch.empty((BATCH_SIZE_PER_CLASS,classes)).float().to(device)
                    y = torch.empty((BATCH_SIZE_PER_CLASS)).long().to(device)

                    for j in range(BATCH_SIZE_PER_CLASS):

                        index = (b*BATCH_SIZE_PER_CLASS) + j

                        x[j,0:3,0:224,0:224] = data_list[c][index][0]
                        y[j] = data_list[c][index][1]

                    pred = model(x)
                    l = loss_fn(pred, y)
                    loss += l


                loss = torch.mean(loss)
                print(loss.tolist())
                history.append(loss.tolist())

                opt.zero_grad()
                loss.backward()
                opt.step()

        return history


    model1 = resnet18(num_classes=classes).to(device)
    model2 = resnet18(num_classes=classes).to(device)
    model3 = resnet18(num_classes=classes).to(device)
    model4 = resnet18(num_classes=classes).to(device)
    model5 = resnet18(num_classes=classes).to(device)

    init_model(model1, nn.init.xavier_uniform_)
    init_model(model2, nn.init.xavier_normal_)
    init_model(model3, nn.init.kaiming_uniform_)
    init_model(model4, nn.init.kaiming_normal_)
    init_model(model5, nn.init.orthogonal_)


    hist1 = train(model1)
    hist2 = train(model2)
    hist3 = train(model3)
    hist4 = train(model4)
    hist5 = train(model5)


    torch.save(model1.state_dict(), "./models/model1.pt")
    torch.save(model2.state_dict(), "./models/model2.pt")
    torch.save(model3.state_dict(), "./models/model3.pt")
    torch.save(model4.state_dict(), "./models/model4.pt")
    torch.save(model5.state_dict(), "./models/model5.pt")


    t = range(len(hist1))

    plt.plot(t,hist1,label="model 1 loss")
    plt.plot(t,hist2,label="model 2 loss")
    plt.plot(t,hist3,label="model 3 loss")
    plt.plot(t,hist4,label="model 4 loss")
    plt.plot(t,hist5,label="model 5 loss")

    plt.xlabel('Batch Number', fontsize=18)
    plt.ylabel('Cross-Entropy Loss', fontsize=18)
    plt.legend()
    plt.show()



    # MC DROPOUT
    # model.eval()
    # enable_dropout(model)
    # with torch.no_grad():
        # print(model(x)[0,0:classes])
        # pass


except Exception as e:

    print('failed with:', str(e))
    if str(device) == "cuda":
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
