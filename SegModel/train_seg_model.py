# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:02 2019

@author: PC
"""

import os
import time
import numpy as np
from net_seg import *
import torch
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy.ndimage.interpolation import rotate
import glob
#from CumulativeAverager import *

def augment(roi, label, ifflip = True, ifrotate=True, ifswap = True):
    if ifrotate:
        angle1 = np.random.rand()*180
        roi = rotate(roi,angle1,axes=(2,3),reshape=False)
        label = rotate(label,angle1,axes=(2,3),reshape=False)
    if ifswap:
        axisorder = np.random.permutation(3)
        roi = np.transpose(roi,np.concatenate([[0],axisorder+1]))
        label = np.transpose(label,np.concatenate([[0],axisorder+1]))
    if ifflip:
        flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
        roi = np.ascontiguousarray(roi[:,::flipid[0],::flipid[1],::flipid[2]])
        label = np.ascontiguousarray(label[:,::flipid[0],::flipid[1],::flipid[2]])
    return roi, label

class GGODataGenerator(Dataset):
    def __init__(self, img_path, phase='train'):
        assert(phase == 'train' or phase == 'val' or phase == 'test')

        self.List_Num = np.array([os.path.split(s)[-1].split('_')[0] for s in glob.glob(img_path+'/*_roi.npy')])
      
        self.img_path = img_path
        self.phase = phase
                
    def __getitem__(self,idx):  
        if self.phase =='train':
            if idx>=len(self.List_Num):
                idx = idx%len(self.List_Num)
                ifflip = True
                ifrotate= False
                ifswap = False
            elif idx>=(len(self.List_Num)*2):
                idx = idx%(len(self.List_Num)*2)
                ifflip = False
                ifrotate= True
                ifswap = False
            elif idx>=(len(self.List_Num)*3):
                idx = idx%(len(self.List_Num)*3)
                ifflip = False
                ifrotate= False
                ifswap = True
            else:
                ifflip = False
                ifrotate= False
                ifswap = False

        if self.phase == 'train':
            dcm_File = self.List_Num[idx]
            roi_path = self.img_path+'/'+dcm_File+'_roi.npy'
            ROI = np.load(roi_path)[np.newaxis,...]
            class_path = self.img_path+'/'+dcm_File+'_label.npy'
            Label = np.load(class_path)[np.newaxis,...]
            ROI, Label = augment(ROI,Label, ifflip = ifflip, ifrotate=ifrotate, ifswap = ifswap)
        else:
            dcm_File = self.List_Num[idx]
            roi_path = self.img_path+'/'+dcm_File+'_roi.npy'
            ROI = np.load(roi_path)
            class_path = self.img_path+'/'+dcm_File+'_label.npy'
            Label = np.load(class_path)
        return ROI, Label
        
    def __len__(self):
        if self.phase == 'train':
            return len(self.List_Num)*4
        else:
            return len(self.List_Num)

def get_lr(epoch, lr):
    if epoch <= epochs * 0.5:
        lr = lr
    elif epoch <= epochs * 0.8:
        lr = 0.1 * lr
    else:
        lr = 0.01 * lr
    return lr

def get_optimizer(st, lr, momentum=0.9):
    if st == 'sgd':
        return torch.optim.SGD(net.parameters(), lr = lr, momentum=momentum)
    elif st == 'adam':
        return torch.optim.Adam(net.parameters(), lr = lr)

def dice_loss(y, pred):
    smooth = 1.

    yflat = y.view(-1)
    predflat = pred.view(-1)
    intersection = (yflat * predflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (yflat.sum() + predflat.sum() + smooth))


def train(dataloader, net, epoch, optimizer, lr, save_freq, save_dir):
    starttime = time.time()
    net.train()

#    lr = get_lr(epoch, 0.001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    metrics = []

    for i, (data,Class) in enumerate(data_loader):
#        print(i)
        data = Variable(data.cuda(async = True))
        target = Variable(Class.cuda(async = True))
        target = target.float()

        output = net(data)
        
#        avg_tool = CumulativeAverager()
        loss_output = dice_loss(output,target)
#        avg_tool.update(loss_output)
        optimizer.zero_grad()   # clear gradients for next train
        loss_output.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        metrics.append(loss_output)

        if epoch % save_freq == 0:            
            state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
                
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))
         
            
    endtime = time.time()
    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('loss %2.4f' % (np.mean(metrics)))
    print('time:%3.2f'%(endtime-starttime))
    
if __name__ == '__main__':
    # torch.cuda.set_device(0)
    net = Unet3D().cuda()
    optimizer = get_optimizer('adam', lr=0.001)
    cudnn.benchmark = True
    img_path = './Nodule_crop'
    dataset = GGODataGenerator(img_path, phase='train')
    data_loader = DataLoader(dataset, batch_size = 12,shuffle = True, num_workers = 0, pin_memory=True)
    save_freq = 20
    epochs = 200
    lr = 0.001
    save_dir = './model'
    for epoch in range(0, epochs + 1):
        train(data_loader, net, epoch, optimizer, lr, save_freq, save_dir)
