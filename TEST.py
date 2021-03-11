from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
from torch.autograd import Variable
from collections import OrderedDict


###Data require
import argparse
from datasets.dataset import VolumeDataset
from datasets.transforms import *
from torch.utils.data import DataLoader

# ###Model require
from models import resnet



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
parser.add_argument('--num_classes', default=4, type=int, help='Number of classes')
parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
parser.set_defaults(no_cuda=False)

parser.add_argument('--device_ids',  type=int, default=1)

# ========================= Model Save ==========================
parser.add_argument('--checkpoint_path', type=str, default='')

args = parser.parse_args()



###main
args.output_file = './TEST.txt'
###load model
model = resnet.resnet18(pretrained=False, num_classes=args.num_classes)
if args.checkpoint_path is '':
    args.checkpoint_path='./pt/epoch10.pt'
model.load_state_dict(torch.load(args.checkpoint_path))
if not args.no_cuda:
    model = model.cuda(args.device_ids)


test_dataset = VolumeDataset(data_root=args.data_root_test, list_file_root=args.list_file_test, modality=args.modality,
    transform=torchvision.transforms.Compose([
        GroupScale((128,128)),
        ToTorchFormatTensor(div=True),
    ]), 
)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=args.num_workers,drop_last=False)

model.eval()
count_correct = 0.
with torch.no_grad():
    for i_batch, sample_batch in enumerate(test_loader):
        Volume = Variable(sample_batch['Volume']).cuda(args.device_ids)
        labels = Variable(sample_batch['label']).long().cuda(args.device_ids)

        Bw,B,outputs = model(Volume)

        _,pred = torch.max(outputs, 1)
        count_correct += torch.sum(pred == labels)
            
        with open(args.output_file, 'a') as out_file:
            out_file.write('labels is:{0}    pred is:{1}\n'.format(labels.data[0].cpu().numpy(),pred.data[0].cpu().numpy()))
            out_file.write('B is:{0}\n'.format(B.data[0].cpu().numpy().tolist()))
            out_file.write('Bw is:{0}\n'.format(Bw.data[0].cpu().numpy().tolist()))
print("Total acc is:",float(count_correct) / len(test_loader.dataset))
