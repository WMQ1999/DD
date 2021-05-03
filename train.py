import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import layers
import loss
import dataset
import sys
from math import sqrt
from params import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_weight(m):
    if isinstance(m, nn.Conv2d):
        o,i,h,w = m.weight.shape
        sc = sqrt(2./(h*w*o))
        m.weight.data = torch.randn(o, i, h, w) * sc

def init_weight_km_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode = 'fan_out')
def train_loop(dataloader, model, loss_fn, optimizer, scheduler = None):
    size = len(dataloader.dataset)

    for batch, dic in enumerate(dataloader):
        x  = dic['x'].to(device)
        z = dic['z'].to(device)
        label = dic['label'].to(device)
        pred = model(x, z)
        # print(x)
        # print(z)
        # print(label)
        # print(pred)
        # print(pred)
        # print(label)
        loss = loss_fn(pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch%10 == 0:
            loss, current = loss.item(), batch * len(x)
            print("loss : {:>7f}  [{:>5d}/{:>5d}]".format(loss, current, size))
        #break
    if scheduler:
        scheduler.step()


# np.set_printoptions(threshold = 10000)
# std = sys.stdout
# outfile = open('out.txt', 'w')
# sys.stdout = outfile

label = loss.create_logloss_label(label_sz, rPos)
# train_dataset = dataset.get_train_dataset(dataset_dir, label)
train_dataset = dataset.get_special_dataset(dataset_dir, label, 'birds')
train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle= True)
model = layers.SiamFuV2()
# model = layers.SiamFC()
# model.load_state_dict(torch.load('siamfc_horse_fuv2.pth'))
#model = model.apply(init_weight)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate_top)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 10 ** (np.log10(learning_rate_bound/learning_rate_top)/epochs), verbose = True)
loss_fn = loss.LogLoss()

for i in range(epochs):
    print(f"Epoch {i+1}\n------------------")
    train_loop(train_loader, model, loss_fn, optimizer, scheduler)
    #break
torch.save(model.state_dict(), 'siamfc_birds_fuv2.pth')
print("DONE!")

# outfile.close()
# sys.stdout = std