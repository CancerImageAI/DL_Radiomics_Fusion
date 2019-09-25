# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:07:39 2019

@author: PC
"""

    
from torch.nn import Module, Sequential 
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid
from torch import nn
import torch

class Conv3D_Block(Module):
        
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):
        
        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
                        Conv3d(inp_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.conv2 = Sequential(
                        Conv3d(out_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())
        
        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        
        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)

def Maxpool3D_Block():
    
    pool = MaxPool3d(kernel_size=2, stride=2, padding=0)
    
    return pool       
        
class Deconv3D_Block(Module):
    
    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        
        super(Deconv3D_Block, self).__init__()
        
        self.deconv = Sequential(
                        ConvTranspose3d(inp_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, output_padding=0, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())
    
    def forward(self, x):
        
        return self.deconv(x)

    
class ClassifyNet(Module):
    def __init__(self, num_feat=[16,32,64,96,128], residual='conv'):
        super(ClassifyNet, self).__init__()
        
        #Encoder process
        self.conv1 = Conv3D_Block(1, num_feat[0], residual=residual)
        self.pool1 = Maxpool3D_Block()
        self.conv2 = Conv3D_Block(num_feat[0], num_feat[1], residual=residual)
        self.pool2 = Maxpool3D_Block()
        self.conv3 = Conv3D_Block(num_feat[1], num_feat[2], residual=residual)
        self.pool3 = Maxpool3D_Block()
        self.conv4 = Conv3D_Block(num_feat[2], num_feat[3], residual=residual)
        self.pool4 = Maxpool3D_Block()
        self.conv5 = Conv3D_Block(num_feat[3], num_feat[4], residual=residual)
        self.conv6 = Conv3D_Block(num_feat[4], num_feat[4], residual=residual)
        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        self.fc1 = nn.Linear(128*4*4*4,128)
        self.fc2 = nn.Linear(128,2)
        self.Relu = nn.ReLU()


    def forward(self, x):
        down_1 = self.conv1(x)
        pool_1 = self.pool1(down_1)
        down_2 = self.conv2(pool_1)
        pool_2 = self.pool2(down_2)
        down_3 = self.conv3(pool_2)
        pool_3 = self.pool3(down_3)
        down_4 = self.conv4(pool_3)
        pool_4 = self.pool4(down_4)
        down_5 = self.conv5(pool_4)        
        conv_6 = self.conv6(down_5)
        conv_6 = self.drop(conv_6)
        view1 = down_5.view(conv_6.size(0),-1)
        fc1 = self.Relu(self.fc1(view1))
        out = self.fc2(fc1)
        
        return out
        
        
        
        
        
    