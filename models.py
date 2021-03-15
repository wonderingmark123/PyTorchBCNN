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
from e2cnn import gspaces
from e2cnn import nn as e2nn


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

# ----------------------------------------------------------------------
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

    def loss(self,FRPred, MiraPred, FRtarget,Miratarget, weights, device="cpu" ):
        """
            Function to calculate weighted 3 term loss function for BCNN
        """
        # if device=="cpu":
        #     y_c1_train = l1_labels(y_train)
        #     y_c2_train = l2_labels(y_train)
        # else:
        #     y_c1_train = l1_labels(y_train.to("cpu")).to(device)
        #     y_c2_train = l2_labels(y_train.to("cpu")).to(device)
        #     weights = weights.to(device)

        l1 = F.cross_entropy( FRPred,FRtarget)
        l2 = F.cross_entropy(MiraPred,Miratarget)
        # l3 = F.cross_entropy(f1, y_train)
        loss = weights[0]*l1 + weights[1]*l2

        return loss
# --------------------------------------------------------------------------
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

class DNSteerableLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5, N=8):
        super(DNSteerableLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        
        self.r2_act = gspaces.FlipRot2dOnR2(N)
        
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        
        out_type = e2nn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])
        self.mask = e2nn.MaskModule(in_type, imsize, margin=1)
        self.conv1 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        in_type = self.pool1.out_type
        out_type = e2nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.conv2 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu2 = e2nn.ReLU(out_type, inplace=True)
        self.pool2 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)
        
        self.gpool = e2nn.GroupPooling(out_type)

        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        
        self.drop  = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
 
    def forward(self, x):
        
        x = e2nn.GeometricTensor(x, self.input_type)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.gpool(x)
        x = x.tensor
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
    
        return x

class DNSteerableMiraSub(nn.Module):
    def __init__(self, 
        in_chan       = 1,
        OutputPara    = [2,5],
        imsize        = 150,
        kernel_size   = 5,
        N             = 16,
        convPara      = [6,16,32,64], 
        FC_FRpara     = [120,84],
        FC_Mirapara   = [120,84]):
        super(DNSteerableMiraSub, self).__init__()
        FRout, FineOUT = OutputPara
        ImsizeFCcoarse = 0.5*(imsize - 2)
        ImsizeFCcoarse = int(0.5*(ImsizeFCcoarse - 2))
        ImsizeFCfine   = int(0.5*(0.5*(ImsizeFCcoarse - 2) - 2))
        self.r2_act = gspaces.FlipRot2dOnR2(N)
        
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        
        out_type = e2nn.FieldType(self.r2_act, convPara[0]*[self.r2_act.regular_repr])
        self.mask = e2nn.MaskModule(in_type, imsize, margin=1)
        self.conv1 = e2nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=1, bias=False)
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        in_type = self.pool1.out_type
        out_type = e2nn.FieldType(self.r2_act, convPara[1]*[self.r2_act.regular_repr])
        self.conv2 = e2nn.R2Conv(in_type, out_type, kernel_size = kernel_size, padding=1, bias=False)
        self.relu2 = e2nn.ReLU(out_type, inplace=True)
        self.pool2 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)
        
        self.gpool1 = e2nn.GroupPooling(out_type)

        self.fc1   = nn.Linear(16*ImsizeFCcoarse*ImsizeFCcoarse, FC_FRpara[0])
        self.fc2   = nn.Linear(FC_FRpara[0], FC_FRpara[1])
        self.fc3   = nn.Linear(FC_FRpara[1], FRout)
        
        self.drop  = nn.Dropout(p=0.5)
        
        in_type = self.pool2.out_type
        out_type = e2nn.FieldType(self.r2_act, convPara[2]*[self.r2_act.regular_repr])
        self.conv3 = e2nn.R2Conv(in_type, out_type, kernel_size = kernel_size, padding=1, bias=False)
        self.relu3 = e2nn.ReLU(out_type, inplace=True)
        self.pool3 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        in_type = self.pool3.out_type
        out_type = e2nn.FieldType(self.r2_act, convPara[3]*[self.r2_act.regular_repr])
        self.conv4 = e2nn.R2Conv(in_type, out_type, kernel_size = kernel_size, padding=1, bias=False)
        self.relu4 = e2nn.ReLU(out_type, inplace=True)
        self.pool4 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)
        self.gpool2 = e2nn.GroupPooling(out_type)

        self.fc4   = nn.Linear(convPara[3]*ImsizeFCfine*ImsizeFCfine, FC_Mirapara[0])
        self.fc5   = nn.Linear(FC_Mirapara[0], FC_Mirapara[1])
        self.fc6   = nn.Linear(FC_Mirapara[1], FineOUT)

        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
    def loss(self,FRPred, MiraPred, FRtarget,Miratarget, weights, device="cpu"):
        """
            same as mira_loss in utils.py 
            Cross_entropy = softmax + log + NLLLoss

            In E2CNNRadGal, softmax was in train function, so log(p) and NLLLoss are needed here.

            (2021/3/13 haotian song)
        """
        # check device for model:
        # device = self.dummy.device
        
        # p : softmax(x)
        # loss_fnc = nn.NLLLoss().to(device=device)
        # loss = loss_fnc(torch.log(p),y)
        l1 = F.cross_entropy( FRPred,FRtarget)
        l2 = F.cross_entropy(MiraPred,Miratarget)
        # l3 = F.cross_entropy(f1, y_train)
        loss = weights[0]*l1 + weights[1]*l2
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
 
    def forward(self, x):
        
        x = e2nn.GeometricTensor(x, self.input_type)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        y = x

        x = self.gpool1(x)
        x = x.tensor
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        Coarse = self.fc3(x)


        x = self.conv3(y)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        x = self.gpool2(x)
        x = x.tensor
        x = x.view(x.size()[0], -1)

        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.drop(x)
        Fine = self.fc6(x)
        
        return Coarse, Fine

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
