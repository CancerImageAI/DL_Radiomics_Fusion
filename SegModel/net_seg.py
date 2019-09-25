# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:07:39 2019

@author: PC
"""

    
from torch.nn import Module, Sequential 
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid
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

    
class Unet3D(Module):
    def __init__(self, num_feat=[16,32,64,96,128], residual='conv'):
        super(Unet3D, self).__init__()
        
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

        
        #Decoder process
        self.upconv4 = Deconv3D_Block(num_feat[4], num_feat[3])
        self.deconv4 = Conv3D_Block(num_feat[3]*2, num_feat[3], residual=residual)
        self.upconv3 = Deconv3D_Block(num_feat[3], num_feat[2])
        self.deconv3 = Conv3D_Block(num_feat[2]*2, num_feat[2], residual=residual)
        self.upconv2 = Deconv3D_Block(num_feat[2], num_feat[1])
        self.deconv2 = Conv3D_Block(num_feat[1]*2, num_feat[1], residual=residual)
        self.upconv1 = Deconv3D_Block(num_feat[1], num_feat[0])
        self.deconv1 = Conv3D_Block(num_feat[0]*2, num_feat[0], residual=residual)
        
        self.out_conv = Conv3d(num_feat[0], 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = Sigmoid()

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
        
        
        up_4 = torch.cat([self.upconv4(down_5), down_4], dim=1)
        deconv_4 = self.deconv4(up_4)
        up_3 = torch.cat([self.upconv3(deconv_4), down_3], dim=1)
        deconv_3 = self.deconv3(up_3)
        up_2 = torch.cat([self.upconv2(deconv_3), down_2], dim=1)
        deconv_2 = self.deconv2(up_2)
        up_1 = torch.cat([self.upconv1(deconv_2), down_1], dim=1)
        deconv_1 = self.deconv1(up_1)
        
        out = self.sigmoid(self.out_conv(deconv_1))
        
        return out
        
        
        
        
        
    
    
    