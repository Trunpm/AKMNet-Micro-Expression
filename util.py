import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import time
import math
from torch.autograd import Variable
import numpy as np



def tr_epoch(model, data_loader, criterion1, criterionB, criterionBw, optimizer, args):
    # training-----------------------------
    model.train()
    loss_value1 = 0.
    loss_valueBw = 0.
    loss_valueB = 0.
    for i_batch, sample_batch in enumerate(data_loader):
        Volume = Variable(sample_batch['Volume']).cuda(args.device_ids)
        labels = Variable(sample_batch['label']).long().cuda(args.device_ids)

        Bw,B,outputs = model(Volume)
        loss1 = criterion1(outputs, labels)
        lossBw = criterionBw(Bw)
        lossB = criterionB(B)
        loss = loss1+lossBw+lossB
        loss_value1 += loss1
        loss_valueBw += lossBw
        loss_valueB += lossB

        if (i_batch+1)>int(np.floor(len(data_loader.dataset)/8))*8:
            loss = loss/(len(data_loader.dataset)-int(np.floor(len(data_loader.dataset)/8))*8)
            loss.backward()
            if (i_batch+1)==len(data_loader.dataset):
                optimizer.step()
                optimizer.zero_grad()
        else:
            loss = loss/8
            loss.backward()
            if (i_batch+1)%8==0:
                optimizer.step()
                optimizer.zero_grad()

    print('epoch Loss1: {:.6f}'.format(float(loss_value1.data)/(i_batch+1)), 'epoch LossBw: {:.6f}'.format(float(loss_valueBw.data)/(i_batch+1)), 'epoch LossB: {:.6f}'.format(float(loss_valueB.data)/(i_batch+1)))
    with open('./logtrain.txt', 'a') as out_file:
        out_file.write('epoch Loss1:{0},epoch LossBw:{1},epoch LossB:{2}'.format(float(loss_value1.data)/(i_batch+1),float(loss_valueBw.data)/(i_batch+1),float(loss_valueB.data)/(i_batch+1))+'\n')

def ts_epoch(model, data_loader, criterion1, criterionB, criterionBw, args):
    model.eval()
    count_correct = 0.
    loss_value1 = 0.
    loss_valueBw = 0.
    loss_valueB = 0.
    with torch.no_grad():
        for i_batch, sample_batch in enumerate(data_loader):
            Volume = Variable(sample_batch['Volume']).cuda(args.device_ids)
            labels = Variable(sample_batch['label']).long().cuda(args.device_ids)

            Bw,B,outputs = model(Volume)

            loss1 = criterion1(outputs, labels)
            lossBw = criterionBw(Bw)
            lossB = criterionB(B)
            loss_value1 += loss1
            loss_valueBw += lossBw
            loss_valueB += lossB

            _,pred = torch.max(outputs, 1)
            count_correct += torch.sum(pred == labels)

        print('Test Loss1: {:.6f}'.format(float(loss_value1.data)/(i_batch+1)),'epoch LossBw: {:.6f}'.format(float(loss_valueBw.data)/(i_batch+1)), 'epoch LossB: {:.6f}'.format(float(loss_valueB.data)/(i_batch+1)))
        print('Acc is:', float(count_correct) / len(data_loader.dataset))
        with open('./logtest.txt', 'a') as out_file:
            out_file.write('Test Loss1:{0},epoch LossBw:{1},epoch LossB:{2},acc is:{3}'.format(float(loss_value1.data)/(i_batch+1),float(loss_valueBw.data)/(i_batch+1),float(loss_valueB.data)/(i_batch+1),float(count_correct) / len(data_loader.dataset))+'\n')

    return float(count_correct) / len(data_loader.dataset)
