# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:44:09 2019

@author: PC
"""

import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor,imageoperations
import os
import pandas as pd
from pandas import DataFrame as DF
import warnings
import time
from time import sleep
from tqdm import tqdm
from skimage import measure


def Img_Normalization(Image_Orig):
    Image_array = sitk.GetArrayFromImage(Image_Orig)
    min_ImgValue = Image_array.min()
    max_ImgValue = Image_array.max()
    ImgRange = max_ImgValue-min_ImgValue
    min_NewValue = 0
    max_NewValue = 1200
    NewRange = max_NewValue-min_NewValue
    Img_array = ((Image_array-min_ImgValue)/ImgRange)*NewRange+min_NewValue
    Img = sitk.GetImageFromArray(Img_array.astype(int))
    Img.SetDirection(Image_Orig.GetDirection())
    Img.SetOrigin(Image_Orig.GetOrigin())
    Img.SetSpacing(Image_Orig.GetSpacing())
#    Img.CopyInformation(Image_Orig)
    return Img
    
def readDCM_Img(FilePath):
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(FilePath)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    return image

def Extract_Features(image,mask,params_path):
    paramsFile = os.path.abspath(params_path)
    extractor = featureextractor.RadiomicsFeaturesExtractor(paramsFile)
    result = extractor.execute(image, mask)
    general_info = {'diagnostics_Configuration_EnabledImageTypes','diagnostics_Configuration_Settings',
                    'diagnostics_Image-interpolated_Maximum','diagnostics_Image-interpolated_Mean',
                    'diagnostics_Image-interpolated_Minimum','diagnostics_Image-interpolated_Size',
                    'diagnostics_Image-interpolated_Spacing','diagnostics_Image-original_Hash',
                    'diagnostics_Image-original_Maximum','diagnostics_Image-original_Mean',
                    'diagnostics_Image-original_Minimum','diagnostics_Image-original_Size',
                    'diagnostics_Image-original_Spacing','diagnostics_Mask-interpolated_BoundingBox',
                    'diagnostics_Mask-interpolated_CenterOfMass','diagnostics_Mask-interpolated_CenterOfMassIndex',
                    'diagnostics_Mask-interpolated_Maximum','diagnostics_Mask-interpolated_Mean',
                    'diagnostics_Mask-interpolated_Minimum','diagnostics_Mask-interpolated_Size',
                    'diagnostics_Mask-interpolated_Spacing','diagnostics_Mask-interpolated_VolumeNum',
                    'diagnostics_Mask-interpolated_VoxelNum','diagnostics_Mask-original_BoundingBox',
                    'diagnostics_Mask-original_CenterOfMass','diagnostics_Mask-original_CenterOfMassIndex',
                    'diagnostics_Mask-original_Hash','diagnostics_Mask-original_Size',
                    'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_VolumeNum',
                    'diagnostics_Mask-original_VoxelNum','diagnostics_Versions_Numpy',
                    'diagnostics_Versions_PyRadiomics','diagnostics_Versions_PyWavelet',
                    'diagnostics_Versions_Python','diagnostics_Versions_SimpleITK',
                    'diagnostics_Image-original_Dimensionality'}
    features = dict((key, value) for key, value in result.items() if key not in general_info)
    feature_info = dict((key, value) for key, value in result.items() if key in general_info)
    return features,feature_info

if __name__ == '__main__':
    start = time.clock()
    warnings.simplefilter('ignore')

    img_path = r'.\GGO_DataSet\test_data\seg_result'
    list_path = r'.\GGO_DataSet\test_data\test.csv'

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
    Label = 0
    
    Feature = []
    for i in tqdm(range(len(List_Num))):
        sleep(0.01)
        dcm_File = List_Num[i]
        roi_path = img_path+'/'+'ROI_'+str(dcm_File)+'.nii'
        ROI = sitk.ReadImage(roi_path)
        
        mask_path = img_path+'/'+'Mask_'+str(dcm_File)+'.nii'
        Mask = sitk.ReadImage(mask_path)
        
        features, feature_info = Extract_Features(ROI, Mask, 'params.yaml')
        if Histopathology[i] == 'AIS' or Histopathology[i] == 'MIA':
            Label = 0
        elif Histopathology[i] == 'IAC':
            Label = 1

        features['Name'] = List_Num[i]
        features['Histopathology'] = Histopathology[i]
        features['Class'] = Label

        Feature.append(features)
    
    df = DF(Feature).fillna('0')
    df.to_csv('./testing_Radiomics_Feature.csv',index=False,sep=',')

    end = time.clock()
    print(end-start)  