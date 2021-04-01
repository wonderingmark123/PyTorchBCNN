

import torch
from torch.utils.data.dataset import Subset
from torchsummary import summary
import os
import numpy as np
from models import BCNN,BCNNmira, DNSteerableMiraSub,ResNetMira
from utils import Conbine_loss, bcnn_loss, load_data_mira,mira_loss,LoadModel_path
from torch.utils.data.dataloader import DataLoader

import matplotlib.pyplot as plt
# -------------------------------------------------------------------------
# History:
# [AMS - 200601] created
# haotian song kaggle 2021/3/21
# -------------------------------------------------------------------------

# parameters

batch_size    = 2                 # number of samples per mini-batch
num_works     = 4                   # Default: 0
PinMemory     = True                # Default: False
imsize        = 150                # image size (original image size is [150,150])
params        = [2,5]             # [coarse1, coarse2]
weightsList   = ([1,0],[0.2,0.8])       # weights for loss  function
TrainLayer    = 1
lr0           = torch.tensor(1e-4)  # speed of convergence ( learning rate)
momentum      = torch.tensor(8e-1)  # momentum for optimizer
decay         = torch.tensor(1e-6)  # weight decay for regularisation
random_seed   = 42
saving_best   = True
Load_epoch    = 199
kernel_size   = 5
epochsList    = (100,200)

Nrot          = 16                  # parameter for DNSteerableLeNet (DNSteerableMiraSub )
frac_val      = 0.2
EnableConLoss = True
LoadingFile   = 'D:/study/PyTorchBCNN/Trained_model/Nrot9DN_kernel5_validation_Res/Modelpara.pth'

# model = DNSteerableMiraSub(OutputPara=params,imsize= imsize,kernel_size=kernel_size,N= Nrot)
model = ResNetMira(kernel_size=kernel_size)
# summary(model, (1, imsize, imsize))
model,epochNow,BCNN_Valaccs,BCNN_testloss,BCNN_trainloss,BCNN_trainaccs = LoadModel_path(model,LoadingFile)

LoadingFile   = 'D:/study/PyTorchBCNN/Trained_model/MiraFully_Weights01.pth'
model,epochNow,Traditional_Valaccs,Traditional_testloss,Traditional_trainloss,Traditional_trainaccs = LoadModel_path(model,LoadingFile)
a = LoadModel_path(model,LoadingFile)

plt.subplot(2,2,1)
plt.plot(BCNN_trainloss)
plt.plot(Traditional_trainloss)
plt.legend(['BCNN Network','Traditional Network'])
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')











plt.show()