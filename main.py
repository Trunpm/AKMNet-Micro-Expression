from __future__ import print_function
from __future__ import division
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.models as models

###Data require
import argparse
from datasets.dataset import VolumeDataset
from datasets.transforms import *
from torch.utils.data import DataLoader

# ###Model require
from models import resnet
from util import tr_epoch, ts_epoch



parser = argparse.ArgumentParser('Resnets')
parser.add_argument('--seed', type=int, default=1)

# ========================= Data Configs ==========================
parser.add_argument('--data_root_train', type=str, default='')
parser.add_argument('--list_file_train', type=str, default='./Train.txt')
parser.add_argument('--data_root_test', type=str, default='')
parser.add_argument('--list_file_test', type=str, default='./Test.txt')
parser.add_argument('--modality', type=str, default='Gray', help='RGB | Gray')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=16)

# ========================= Model Configs ==========================
parser.add_argument('--premodel', default='XXX/epoch100.pt', type=str, help='Pretrained model (.pth)')
parser.add_argument('--num_classes', default=4, type=int, help='Number of classes')
parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
parser.set_defaults(no_cuda=False)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--PenaltyBw', type=float, default=1)
parser.add_argument('--PenaltyB', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--device_ids',  type=int, default=1)

# ========================= Model Save ==========================
parser.add_argument('--save_path', type=str, default='./pt')
parser.add_argument('--checkpoint_path', type=str, default='')

args = parser.parse_args()



###Data read
train_dataset = VolumeDataset(data_root=args.data_root_train, list_file_root=args.list_file_train, modality=args.modality,
                            transform=torchvision.transforms.Compose([
                                GroupScaleRandomCrop((144,144),(128,128)),
                                ToTorchFormatTensor(div=True),
                            ]),
)
train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,drop_last=False)

test_dataset = VolumeDataset(data_root=args.data_root_test, list_file_root=args.list_file_test, modality=args.modality,
                            transform=torchvision.transforms.Compose([
                                GroupScale((128,128)),
                                ToTorchFormatTensor(div=True),
                            ]), 
)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=args.num_workers,drop_last=False)


# # ###Model
model = resnet.resnet18(pretrained=False, num_classes=args.num_classes)
if args.premodel:
    pretrained_dict = torch.load(args.premodel) 
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:',missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
if not args.no_cuda:
    model = model.cuda(args.device_ids)
print(model)


# ###Hyperparam
criterion1 = nn.CrossEntropyLoss()
if not args.no_cuda:
    criterion1 = criterion1.cuda(args.device_ids)
optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch/5, eta_min=1e-8, last_epoch=-1)

class BwLoss(nn.Module):    
    def __init__(self):        
        super(BwLoss, self).__init__()          
    def forward(self, Bw):
        loss_Bw = 0.0
        for i in range(Bw.shape[0]):
            temp = Bw[i,:]
            loss_Bw += 2.0-(torch.mean(temp[temp>torch.mean(temp)])-torch.mean(temp[temp<torch.mean(temp)]))
        return  args.PenaltyBw*loss_Bw/Bw.shape[0]
criterionBw = BwLoss()
if not args.no_cuda:
    criterionBw = criterionBw.cuda(args.device_ids)

class BLoss(nn.Module):    
    def __init__(self):        
        super(BLoss, self).__init__()          
    def forward(self, B):
        loss_B = 0.0
        for i in range(B.shape[0]):
            loss_B += torch.max(torch.Tensor([0.0]).cuda(args.device_ids), torch.sum(B[i,:]) - torch.Tensor([1.0]).cuda(args.device_ids))     
        return  args.PenaltyB*loss_B/B.shape[0]
criterionB = BLoss()
if not args.no_cuda:
    criterionB = criterionB.cuda(args.device_ids)

Acc_best = 0.0
for epoch in range(1,args.epoch+1):
    print('epoch {}:'.format(epoch), 'lr is {}'.format(optimizer.param_groups[0]['lr']))
    ###train and test
    print("Training-------------------")
    tr_epoch(model=model, data_loader=train_loader, criterion1=criterion1, criterionB=criterionB, criterionBw=criterionBw, optimizer=optimizer, args=args)
    print("Testing====================")
    Acc = ts_epoch(model=model, data_loader=test_loader, criterion1=criterion1, criterionB=criterionB, criterionBw=criterionBw, args=args)
    scheduler.step(epoch)
    #save model
    if epoch>=10:
        if Acc>=Acc_best:
            Acc_best = Acc
            torch.save(model.state_dict(), args.save_path + '/' + 'epoch' + str(epoch) + '.pt')