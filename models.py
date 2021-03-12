# -------------------------------------------------------------------------
# History:
# [AMS - 200601] created
# [AMS - 200601] added lenet for comparison
# [AMS - 200601] added BaseCNN for comparison
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from layers import CoarseFR, ConvBlock, CoarseBlock,ConvUnit


class BCNN(nn.Module):
    def __init__(self, in_chan, params, kernel_size=3,imsize=50):
        super(BCNN, self).__init__()

        c1_targets, c2_targets, out_chan = params
        imsize4 = int(imsize/4)
        imsize8 = int(imsize4/2)
        imsize16 = int(imsize8/2)
        self.convblock1 = ConvBlock(in_channels=in_chan, hidden=32, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, hidden=128, out_channels=128)
        self.coarse1    = CoarseBlock(in_features=128*imsize4*imsize4, hidden=128, out_features=c1_targets)
        
        self.convblock3 = ConvBlock(in_channels=128, hidden=256, out_channels=256)
        self.coarse2    = CoarseBlock(in_features=256*imsize8*imsize8, hidden=1024, out_features=c2_targets)
        
        self.convblock4 = ConvBlock(in_channels=256, hidden=512, out_channels=512)
        self.coarse3    = CoarseBlock(in_features=512*imsize16*imsize16, hidden=1024, out_features=out_chan)
        # self.coarse1    = CoarseBlock(in_features=128*12*12, hidden=128, out_features=c1_targets)
        # self.coarse2    = CoarseBlock(in_features=256*6*6, hidden=1024, out_features=c2_targets)
        # self.coarse3    = CoarseBlock(in_features=512*3*3, hidden=1024, out_features=out_chan)


    def forward(self, x):
        # x [batch 1 70 70]
        x = self.convblock1(x)
        # x [batch 64 35 35]
        x = self.convblock2(x)
        # x [batch 128 17 17]

        l1 = x.view(x.size()[0], -1)
        c1, c1_pred = self.coarse1(l1)

        x = self.convblock3(x)

        l2 = x.view(x.size()[0], -1)
        c2, c2_pred = self.coarse2(l2)

        x = self.convblock4(x)

        l3 = x.view(x.size()[0], -1)
        f1, f1_pred = self.coarse3(l3)

        return c1, c2, f1

# -----------------------------------------------------------------------------
class BCNNmira(nn.Module):
    def __init__(self, in_chan, OutputParameters, 
    kernel_size = 5,
    imsize      = 150,
    convPara    = [6,16,32,64] , 
    FC_FRpara   = [120,84],
    FC_Mirapara = [120,84]):
        super(BCNNmira, self).__init__()

        c1_targets, c2_targets = OutputParameters
        if kernel_size ==5:
            imsize4 = int(imsize/4)-1
        else:
            imsize4 = int(imsize/4)
        imsize8 = int(imsize4/2)
        if kernel_size == 5:
            imsize16 = int(imsize8/2)-2
        else:
            imsize16 = int(imsize8/2)
        self.convunit1 = ConvUnit(in_channels=in_chan,out_channels=convPara[0],kernel_size=kernel_size)
        self.convunit2 = ConvUnit(in_channels=convPara[0],out_channels=convPara[1],kernel_size=kernel_size)
        self.coarse1   = CoarseFR(in_features=convPara[1]*imsize4*imsize4, hidden1=FC_FRpara[0],hidden2=FC_FRpara[1], out_features=c1_targets)
        self.convunit3 = ConvUnit(in_channels=convPara[1],out_channels=convPara[2],kernel_size=kernel_size)
        self.convunit4 = ConvUnit(in_channels=convPara[2],out_channels=convPara[3],kernel_size=kernel_size)
        

        self.coarse2    = CoarseFR(in_features=convPara[3]*imsize16*imsize16, hidden1=FC_Mirapara[0],hidden2=FC_Mirapara[1], out_features=c2_targets)


        # imsize4 = int(imsize/4)
        # imsize8 = int(imsize4/2)
        # imsize16 = int(imsize8/2)
        # self.convblock1 = ConvBlock(in_channels=in_chan, hidden=32, out_channels=64)
        # self.convblock2 = ConvBlock(in_channels=64, hidden=128, out_channels=128)
        # self.coarse1    = CoarseBlock(in_features=128*imsize4*imsize4, hidden=128, out_features=c1_targets)
        
        # self.convblock3 = ConvBlock(in_channels=128, hidden=256, out_channels=256)
        # self.coarse2    = CoarseBlock(in_features=256*imsize8*imsize8, hidden=1024, out_features=c2_targets)
        
        # self.convblock4 = ConvBlock(in_channels=256, hidden=512, out_channels=512)
        # self.coarse3    = CoarseBlock(in_features=512*imsize16*imsize16, hidden=1024, out_features=out_chan)
        # self.coarse1    = CoarseBlock(in_features=128*12*12, hidden=128, out_features=c1_targets)
        # self.coarse2    = CoarseBlock(in_features=256*6*6, hidden=1024, out_features=c2_targets)
        # self.coarse3    = CoarseBlock(in_features=512*3*3, hidden=1024, out_features=out_chan)


    def forward(self, x):

        x = self.convunit1(x)
        x = self.convunit2(x)

        l1 = x.view(x.size()[0], -1)
        c1, c1_softmax = self.coarse1(l1)

        x = self.convunit3(x)
        x = self.convunit4(x)

        l2 = x.view(x.size()[0], -1)
        c2, c2_softmax = self.coarse2(l2)

        # x = self.convblock4(x)

        # l3 = x.view(x.size()[0], -1)
        # f1, f1_softmax = self.coarse3(l3)

        # ----------------------------------------
        # # x [batch 1 70 70]
        # x = self.convblock1(x)
        # # x [batch 64 35 35]
        # x = self.convblock2(x)
        # # x [batch 128 17 17]

        # l1 = x.view(x.size()[0], -1)
        # c1, c1_softmax = self.coarse1(l1)

        # x = self.convblock3(x)

        # l2 = x.view(x.size()[0], -1)
        # c2, c2_softmax = self.coarse2(l2)

        # # x = self.convblock4(x)

        # # l3 = x.view(x.size()[0], -1)
        # # f1, f1_softmax = self.coarse3(l3)




        return c1, c2

# ---------------------------------------------------------------------------
class BaseCNN(nn.Module):
    def __init__(self, in_chan, params, kernel_size=3):
        super(BaseCNN, self).__init__()

        c1_targets, c2_targets, out_chan = params

        self.convblock1 = ConvBlock(in_channels=in_chan, hidden=32, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, hidden=128, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, hidden=256, out_channels=256)
        self.convblock4 = ConvBlock(in_channels=256, hidden=512, out_channels=512)
        self.coarse3    = CoarseBlock(in_features=512*3*3, hidden=1024, out_features=out_chan)

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(x.size()[0], -1)
        f1, f1_pred = self.coarse3(x)

        return f1, f1_pred


# -----------------------------------------------------------------------------


class LeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5):
        super(LeNet, self).__init__()

        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))

        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        self.drop  = nn.Dropout(p=0.5)

        self.init_weights()

    def init_weights(self):
        # weight initialisation:
        # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Linear):
                y = m.in_features
                nn.init.uniform_(m.weight, -np.sqrt(3./y), np.sqrt(3./y))
                nn.init.constant_(m.bias, 0)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x
