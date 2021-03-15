# -------------------------------------------------------------------------
# History:
# [AMS - 200601] created
# -------------------------------------------------------------------------

import torch
from torch.utils.data.dataset import Subset
from torchsummary import summary
import os
import numpy as np
from models import BCNN,BCNNmira, DNSteerableMiraSub
from utils import bcnn_loss, load_data_mira,mira_loss
from torch.utils.data.dataloader import DataLoader

# parameters

batch_size    = 2                 # number of samples per mini-batch
num_works     = 4                   # Default: 0
PinMemory     = True                # Default: False
imsize        = 150                # image size (original image size is [150,150])
params        = [2,5]             # [coarse1, coarse2]
weightsList   = ([0.9,0.1],[0.2,0.8])       # weights for loss  function
TrainLayer    = 0
lr0           = torch.tensor(1e-4)  # speed of convergence ( learning rate)
momentum      = torch.tensor(8e-1)  # momentum for optimizer
decay         = torch.tensor(1e-6)  # weight decay for regularisation
random_seed   = 42
saving_best   = True
Load_epoch    = 0
kernel_size   = 5
epochsList    = (100,200)
SaveModelFile = 'D:\\study\\PyTorchBCNN\\Trained_model\\Nrot9DN_kernel5_validation'
Nrot          = 16                  # parameter for DNSteerableLeNet (DNSteerableMiraSub )
frac_val      = 0.2
# -----------------------------------------------------------------------------
def SavingModel(model,optimizer,epoch,MINloss,epoch_testaccs,epoch_testloss,epoch_trainaccs,epoch_trainloss):
    if not os.path.isdir(SaveModelFile):
        os.mkdir(SaveModelFile)
    SaveModelFileNow = os.path.join(SaveModelFile,str(epoch))
    if not os.path.isdir(SaveModelFileNow):
        os.mkdir(SaveModelFileNow)
    state = {'net'      : model.state_dict(), 
        'optimizer'     : optimizer.state_dict(),
        'epoch'         : epoch,
        'batch_size'    : batch_size   ,              # number of samples per mini-batch
        'imsize'        : imsize,
        'momentum'      : momentum.numpy  ,# momentum for optimizer
        'decay'         : decay.numpy  ,# weight decay for regularisation
        'random_seed'   : random_seed,
        'kernel_size'   : kernel_size, # kenel_size for conv layers    
        'saving_best'   : saving_best,
        'SaveModelFile' : SaveModelFile,
        'MINloss'       : MINloss,
        'epoch_testaccs': epoch_testaccs,
        'epoch_testloss': epoch_testloss,
        'epoch_trainaccs':epoch_trainaccs,
        'epoch_trainloss':epoch_trainloss,

      }
    torch.save(state,os.path.join(SaveModelFileNow,'Modelpara.pth'))
def LoadModel(model,epoch):
    
    SaveModelFileNow = os.path.join(SaveModelFile,str(epoch))
    if not os.path.isdir(SaveModelFileNow):
        return model,0,0,[],[],[],[]
    state  = torch.load(os.path.join(SaveModelFileNow,'Modelpara.pth'))
    model.load_state_dict(state['net'])
    epoch               = state['epoch']
    MINloss             = state['MINloss']
    epoch_testaccs      = state['epoch_testaccs']
    epoch_testloss      = state['epoch_testloss']
    epoch_trainloss     = state['epoch_trainloss']
    epoch_trainaccs     = state['epoch_trainaccs']
    return model,epoch,epoch_testaccs,epoch_testloss,epoch_trainloss,epoch_trainaccs
# -----------------------------------------------------------------------------
def main():
    
    epochs   = epochsList[TrainLayer]
    weights  = weightsList[TrainLayer]
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ",device)
    weights = torch.from_numpy(np.array(weights)).to(device)
    print('weights are {}'.format(weights))
    # -----------------------------------------------------------------------------
    trainset = load_data_mira(train=True,imsize = imsize)
    if frac_val>0.:
        dataset_size = len(trainset)
        nval = int(frac_val*dataset_size)

        indices = list(range(dataset_size))
        train_indices, val_indices = indices[nval:], indices[:nval]
        
        train_sampler = Subset(trainset, train_indices)
        valid_sampler = Subset(trainset, val_indices)

        trainloader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size, shuffle=True,num_workers=num_works,pin_memory=PinMemory)
        testloader = torch.utils.data.DataLoader(valid_sampler, batch_size=batch_size, shuffle=True,num_workers=num_works,pin_memory=PinMemory)
    else:
        # setting frac_val to zero will use the test set for validation
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers=num_works,pin_memory=PinMemory)

        testset = load_data_mira(train=False,imsize = imsize)

        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,num_workers=num_works,pin_memory=PinMemory)


    # -----------------------------------------------------------------------------

    # model = BCNNmira(in_chan=1, OutputParameters=params, kernel_size= kernel_size ,imsize = imsize)
    model = DNSteerableMiraSub(OutputPara=params,imsize= imsize,kernel_size=kernel_size,N= Nrot)
    learning_rate = lr0
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=2, factor=0.9)


    minTestAccs = 0 # if the test accuracy is higher than this, save the current model
    # -----------------------------------------------------------------------------

    summary(model, (1, imsize, imsize))
    model = model.to(device)

    # -----------------------------------------------------------------------------


    epoch_trainaccs, epoch_testaccs = [], []
    epoch_trainloss, epoch_testloss = [], []

    if Load_epoch > 0:
        model,epochNow,epoch_testaccs,epoch_testloss,epoch_trainloss,epoch_trainaccs =LoadModel(model,Load_epoch)

    for epoch in range(Load_epoch,epochs):
        # ----------------------------------------------------------
        #                   training process
        # ----------------------------------------------------------
        model.train()
        train_losses, train_accs = [], []; acc = 0
        for batch, (MiraImage, MiraLabel, FRLabel) in enumerate(trainloader):
            # optimizer.zero_grad()
            model.zero_grad()
            MiraImage, MiraLabel,FRLabel = MiraImage.to(device), MiraLabel.to(device),FRLabel.to(device)
            FRPred, MiraPred = model(MiraImage)

            loss = mira_loss(FRPred, MiraPred,FRLabel,MiraLabel, weights, device=device)
            loss.backward()
            optimizer.step()

            acc = (MiraPred.argmax(dim=-1) == MiraLabel).to(torch.float32).mean()
            train_accs.append(acc.mean().item())
            train_losses.append(loss.item() * MiraImage.size(0))

        print('Train: {}, Loss: {}, Accuracy: {}'.format(epoch, np.mean(train_losses), np.mean(train_accs)))

        with torch.no_grad():
            # -----------------------------------------------------
            #                   Testing process
            # -------------------------------------------------------
            model.eval()
            test_losses, test_accs = [], []; acc = 0
            for i, (MiraImageTest, MiraLabelTest,FRLabelTest) in enumerate(testloader):

                MiraImageTest, MiraLabelTest,FRLabelTest = MiraImageTest.to(device), MiraLabelTest.to(device),FRLabelTest.to(device)
                FRLabelTestPre, MiraLabelTestPre = model(MiraImageTest)

                loss = mira_loss(FRLabelTestPre, MiraLabelTestPre,FRLabelTest, MiraLabelTest, weights, device=device)
                if TrainLayer == 1 :
                    acc = (MiraLabelTestPre.argmax(dim=-1) == MiraLabelTest).to(torch.float32).mean()
                if TrainLayer == 0 :
                    acc = (FRLabelTestPre.argmax(dim=-1) == FRLabelTest).to(torch.float32).mean()
                test_losses.append(loss.item())
                test_accs.append(acc.mean().item())
        scheduler.step(np.mean(test_losses))
        print(' Test: {}, Loss: {}, Accuracy: {}'.format(epoch, np.mean(test_losses), np.mean(test_accs)))
        print('Current learning rate is: {}'.format(optimizer.param_groups[0]['lr']))
        print('---------------------------------------')
        epoch_trainaccs.append(np.mean(train_accs))
        epoch_testaccs.append(np.mean(test_accs))
        epoch_trainloss.append(np.mean(train_losses))
        epoch_testloss.append(np.mean(test_losses))
        
        if np.mean(test_accs) > minTestAccs and saving_best and epoch > 10:
            minTestAccs = np.mean(test_accs)
            epochNow = epoch
            print( 'Saving model ...' )
            SavingModel(model,optimizer,epoch,minTestAccs,epoch_testaccs,epoch_testloss,epoch_trainaccs,epoch_trainloss)
    SavingModel(model,optimizer,epoch,minTestAccs,epoch_testaccs,epoch_testloss,epoch_trainaccs,epoch_trainloss)
    print("Final test error: ",100.*(1 - epoch_testaccs[-1]))
    print("Best model Testing accuracy: ",100.*minTestAccs)
    print("Best model Testing epoch: ",epochNow)

if __name__ == '__main__':
    main()