# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:02:14 2019

@author: PC
"""

from lidcxmlparser import *
import pydicom as dicom
import numpy as np
import os
import glob
from skimage import draw, measure
import scipy

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(s) for s in glob.glob(path+'/*.dcm')]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

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
    initial_seed = [seed_pos[0], seed_pos[1], seed_pos[2]]
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
def get_nodule_center(xml_path, image, slice_loc):
    gt = LIDCXmlParser(xml_path)
    gt.parse()
    mask = np.zeros(image.shape)
    for indx, rad in enumerate(gt.rad_annotations): #has 4 radiologistes
        mask_1 = np.zeros(image.shape)
        for nod in rad.nodules: #nod is one NormalNodule
            # if nod.characterstics.texture <= 3:                
            for nod_roi in nod.rois:  
                z_index = np.where(slice_loc==nod_roi.z)[0][0]
                xy = np.array(nod_roi.roi_xy)
                xx, yy = draw.polygon(xy[:,1],xy[:,0])
                mask_1[xx,yy,z_index] = 1
        mask = mask+mask_1
    mask = np.array(mask>1).astype(int)
    L_mask = measure.label(mask)
    L_props= measure.regionprops(L_mask)
    center_pos = []
    for props in L_props:
        center = np.array(props.centroid).astype(int)
        center_pos.append(center)
    return center_pos,mask

  
def search_xml(file_dir):  
    xml_path=[]  
    for root, dirs, files in os.walk(file_dir): 
        for file in files: 
            if os.path.splitext(file)[1] == '.xml': 
                xml_path.append(os.path.join(root, file)) 
    return xml_path

if __name__ == '__main__':
    
    home_path = './LIDC'
    save_path = './Nodule_crop'
    data_list = os.listdir(home_path)
    Nodule_num = 0
    for patient_path in data_list:
        img_path = os.path.join(home_path,patient_path)
        xml_listpath = search_xml(img_path)
        for xml_path in xml_listpath:
            path, xml_f = os.path.split(xml_path)
            dcm = [s for s in glob.glob(path+'/*.dcm')]
            if len(dcm) > 10:
                dicom_slices = load_scan(path)
                image = [s.pixel_array*int(s.RescaleSlope)+int(s.RescaleIntercept) for s in dicom_slices ]
                image = np.array(image).transpose(1,2,0)
                slice_loc = np.array([s.ImagePositionPatient[2] for s in dicom_slices]).astype(float)
                spacing = np.array([dicom_slices[0].PixelSpacing[0], 
                                    dicom_slices[0].PixelSpacing[1],
                                    dicom_slices[0].SliceThickness]).astype(float)
                center_pos,mask = get_nodule_center(xml_path, image, slice_loc)
                if len(center_pos) != 0:
                    image_new = {}
                    image_new['array'] = image
                    image_new['Spacing'] = spacing
                    img, img_size, resize_factor = resample(image_new)
                    mask_new = {}
                    mask_new['array'] = mask
                    mask_new['Spacing'] = spacing
                    label, label_size, resize_factor = resample(mask_new)
                    img = normalize_hu(img)   
                    for pos in center_pos:
                        Nodule_num = Nodule_num+1
                        seed_pos = [pos[0], pos[1], pos[2]]
                        ROI = crop_roi(img, img_size, seed_pos, [64,64,64] , resize_factor)
                        ROI_label = crop_roi(label, label_size, seed_pos, [64, 64, 64], resize_factor)
                        ROI = (ROI.astype(np.float32)-128)/128
#
                        np.save(os.path.join(save_path,str(Nodule_num)+'_roi.npy'), ROI)
                        np.save(os.path.join(save_path,str(Nodule_num)+'_label.npy'), ROI_label)
                    
        


        
    
