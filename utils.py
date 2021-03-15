# -------------------------------------------------------------------------
# History:
# [AMS - 200601] created
# -------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MiraBest import MBFRConfident, MBFRConfidentSub
from PIL import Image
import torchvision.transforms as transforms

#  -----------------------------------------------
# the following part is for MiraBest dataset
# -------------------------------------------------

def train(model, trainloader, optimizer, device):
    
    train_loss = 0.0

    model.train()
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()

        p_y = F.softmax(model(data), dim=1)
        loss = model.loss(p_y, labels)
            
        train_loss += loss.item() * data.size(0)

        loss.backward()
        optimizer.step()

    train_loss /= len(trainloader.dataset)
    return train_loss


def test(model, testloader, device):
    
    correct = 0
    total = 0
    test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            data, labels = data.to(device), labels.to(device)

            p_y = F.softmax(model(data), dim=1)
            loss = model.loss(p_y, labels)
                
            test_loss += loss.item() * data.size(0)

            preds = p_y.argmax(dim=1, keepdim=True)
            correct += preds.eq(labels.view_as(preds)).sum().item()

        test_loss /= len(testloader.dataset)
        accuracy = correct / len(testloader.dataset)

    return test_loss, accuracy


def load_data_mira(dataDir = 'mirabest', train=True ,imsize = 150):
    """
    Load miraBest data from dataDir. If that doesn't exist, download it.

    imsize: 150                      # pixels on side of image
    datamean: 0.0031
    datastd: 0.0350
    """
    
    # -----------------------------------------------------------------------------
    # Data loading:
    datamean = 0.0031
    datastd = 0.0350
    
    crop        = transforms.CenterCrop(imsize)
    pad         = transforms.Pad((0, 0, 1, 1), fill=0)
    totensor    = transforms.ToTensor()
    normalise   = transforms.Normalize(datamean , datastd )
    # transform = transforms.Compose([
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.Resize(50),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,))
    #         ])
    
    transform = transforms.Compose([
        crop,
        pad,
        transforms.RandomRotation(360, resample=Image.BILINEAR, expand=False),
        totensor,
        normalise,
    ])


    train_data = MBFRConfidentSub(dataDir, train=train, download=True, transform=transform)
    return train_data

def mira_loss(FRPred, MiraPred, FRtarget,Miratarget, weights, device="cpu" ):
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



#  -----------------------------------------------
# the following part is for CIFAR5 classification 
# -------------------------------------------------
def bcnn_loss(c1, c2, f1, y_train, weights, device="cpu"):
    """
        Function to calculate weighted 3 term loss function for BCNN
    """

    if device=="cpu":
      y_c1_train = l1_labels(y_train)
      y_c2_train = l2_labels(y_train)
    else:
      y_c1_train = l1_labels(y_train.to("cpu")).to(device)
      y_c2_train = l2_labels(y_train.to("cpu")).to(device)
      weights = weights.to(device)

    l1 = F.cross_entropy(c1, y_c1_train)
    l2 = F.cross_entropy(c2, y_c2_train)
    l3 = F.cross_entropy(f1, y_train)

    loss = weights[0]*l1 + weights[1]*l2 + weights[2]*l3

    return loss
def l1_labels(labels):

    """
        0: vehicle (0:plane, 1:car, 4:truck)
        1: animal (2:bird, 3:horse)
    """

    l1_labels = np.zeros(labels.size())
    l1_labels[np.where(labels==2)]=1
    l1_labels[np.where(labels==3)]=1

    return torch.tensor(l1_labels, dtype=torch.long)


def l2_labels(labels):

    """
        0: air (0:plane)
        1: ground (1:car, 4:truck)
        2: bird (2:bird)
        3: horse (3:horse)
    """

    l2_labels = np.zeros(labels.size())
    l2_labels[np.where(labels==1)]=1
    l2_labels[np.where(labels==2)]=2
    l2_labels[np.where(labels==3)]=3
    l2_labels[np.where(labels==4)]=1

    return torch.tensor(l2_labels, dtype=torch.long)
