# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:38:04 2019

@author: PC
"""

import os
import time
import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy.ndimage.interpolation import rotate
import pandas as pd
import SimpleITK as sitk
import scipy
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm

def readDCM_Img(FilePath):
    img = {}
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(FilePath)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    img_array = sitk.GetArrayFromImage(image) # z,y,x
    Spacing = image.GetSpacing()
#    Origin = image.GetOrigin()
    img_array = img_array.transpose(2,1,0)#x,y,z
    img['array'] = img_array
    img['Spacing'] = np.array(Spacing).astype(float)
#    img['Origin'] = Origin
    return img

def resample(img, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    image = img['array']
    spacing = img['Spacing']
    img_size = np.array(image.shape)
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, img_size, real_resize_factor

def normalize_hu(image):
    MIN_BOUND = -1200.0
    MAX_BOUND = 600.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1
    image[image < 0] = 0
    image = (image*255).astype('uint8')
    return image

def crop_roi(resampled_img, img_size, seed_pos, crop_size, resize_factor):
    initial_seed = [seed_pos[0], seed_pos[1], img_size[2]-seed_pos[2]]
    trans_seed = initial_seed*resize_factor
    start = []
    end= []
    for i in range(3):
        s = np.floor(trans_seed[i]-(crop_size[i]/2))
        e = np.ceil(trans_seed[i]+(crop_size[i]/2))
        if s<0:
            s = 0
        if e>resampled_img.shape[i]:
            e = resampled_img.shape[i]
        if e-s != crop_size[i]:
            pad = e-s-crop_size[i]
            if s==0:
                e = e-pad
            else:
                s = s+pad
        start.append(int(s))
        end.append(int(e))       
#    print(start,end,pad)
    roi = resampled_img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]      
        
    return roi
def save_img(image, outputImageFileName):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outputImageFileName)
    writer.Execute(image)

if __name__ == "__main__":
    img_path = r'.\GGO_DataSet\test_data\DCM'
    list_path = r'.\GGO_DataSet\test_data\test.csv'
    save_dir = r'.\GGO_DataSet\test_data\test_Img'
    
    f = open(list_path)
    GGO_list = pd.read_csv(f)
    List_Num = np.array(GGO_list['Num'].tolist())
    Type = GGO_list['Type'].tolist()
    Class = np.array(GGO_list['Class'].tolist()) 
    List_Num = List_Num[[i for i,x in enumerate(Type) if x!='Solid']]
    Class= Class[[i for i,x in enumerate(Type) if x!='Solid']]
    List_Num = List_Num[[i for i,x in enumerate(Class) if x!=0]]
    Histopathology = np.array(GGO_list['Histopathology'].tolist())
    Histopathology = Histopathology[[i for i,x in enumerate(Type) if x!='Solid']]
    Histopathology = Histopathology[[i for i,x in enumerate(Class) if x!=0]]
    x_list = np.array(GGO_list['X'].tolist())
    x_list = x_list[[i for i,x in enumerate(Type) if x!='Solid']]
    x_list = x_list[[i for i,x in enumerate(Class) if x!=0]]
    y_list = np.array(GGO_list['Y'].tolist())
    y_list = y_list[[i for i,x in enumerate(Type) if x!='Solid']]
    y_list = y_list[[i for i,x in enumerate(Class) if x!=0]]
    z_list = np.array(GGO_list['Z'].tolist())
    z_list = z_list[[i for i,x in enumerate(Type) if x!='Solid']]
    z_list = z_list[[i for i,x in enumerate(Class) if x!=0]]
    prob = []
    prob_label = []
    real_class = []
    test_result=[]
    for i in tqdm(range(len(List_Num))): 
        name_split = List_Num[i].split('_')
        dcm_File = name_split[0]+'_'+name_split[1]
        img_fold = os.path.join(img_path, dcm_File)
        img = readDCM_Img(img_fold)
        image, img_size, resize_factor = resample(img)
        X = round(x_list[i])
        Y = round(y_list[i])
        Z = round(z_list[i])
        image = normalize_hu(image)
        seed_pos = [X, Y, Z]
        ROI = crop_roi(image, img_size, seed_pos, [64,64,64] , resize_factor)
        new_spacing = [1,1,1]
        ROI_sitk = sitk.GetImageFromArray(ROI)
        ROI_sitk.SetSpacing(new_spacing)
        ROI = ROI[np.newaxis,...]
        ROI = (ROI.astype(np.float32)-128)/128 
        
        if Histopathology[i] == 'IAC':
            GGO_Class = 1
        else:
            GGO_Class = 0
        np.save(os.path.join(save_dir,List_Num[i]+'_roi'),ROI)
        np.save(os.path.join(save_dir,List_Num[i]+'_label'),np.array(GGO_Class))

