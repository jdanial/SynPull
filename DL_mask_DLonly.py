# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:31:53 2023

@author: Jasmine Analysis PC
"""

import numpy as np
import os
from skimage import io, filters
from skimage.morphology import disk, erosion, dilation, white_tophat, reconstruction
from skimage.measure import label, regionprops
import pandas as pd
from PIL import Image
from tifffile import imsave
from scipy.stats import norm
import cv2


DL_path = r'E:\Synaptosomes_080923_DL\Red 0.7 to 2.2'

size_upper = 2200 # nm, maximum diameter of the dot
size_lower = 700 # nm, minimum diameter of the dot

px_size = 105 # pixel size of the image
SNR_thres = 2

def getListFiles(path, kwd = ''):
    filelist = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            if kwd in filespath:
                filelist.append(os.path.join(root,filespath)) 
    return filelist

def get_mask(img_name, size_upper, size_lower, px_size, SNR_thres):
    
    im = io.imread(img_name)
    if len(im.shape)==3:
        im = np.mean(im, axis = 0)
    
    mu,sigma = norm.fit(im)
    thres_img = mu + SNR_thres*sigma
    mask = np.where(im>thres_img, 1, 0)
    mask_e = erosion(mask, disk(2))
    mask_d = dilation(mask_e, disk(2))
    mask_label = label(mask_d)
    props = regionprops(mask_label)
    upper_d = size_upper/px_size
    lower_d = size_lower/px_size
    bad_dot_label = []
    for j in props:
        d = j.axis_major_length
        if d > upper_d or d<lower_d:
            bad_dot_label.append(j.label)
    for l in bad_dot_label:
        mask_label[mask_label==l] = 0
    mask_label[mask_label>0] = 1
    mask_binary = mask_label.copy()
    mask_labeled = label(mask_binary)
    
    mask_props = regionprops(mask_labeled)
    mask_props = [[reg.label-1,
                   reg.centroid[1],
                   reg.centroid[0],
                   reg.area, 
                   reg.perimeter,
                   4*np.pi*reg.area/(reg.perimeter**2),
                   reg.axis_major_length,
                   reg.axis_minor_length,
                   reg.eccentricity,
                   reg.solidity] for reg in mask_props]
    if len(mask_props)!=0:
        mask_props_df = pd.DataFrame(mask_props)
        mask_props_df.columns = ['ID', 'Centroid_X', 'Centroid_Y', 'Area', 
                                 'Perimeter', 'Circularity', 'Maj_axis_len', 
                                 'Min_axis_len', 'Eccentricity', 'Solidity']
    else:
        mask_props_df = pd.DataFrame(np.zeros((1, 10)))
        mask_props_df.columns = ['ID', 'Centroid_X', 'Centroid_Y', 'Area', 
                                 'Perimeter', 'Circularity', 'Maj_axis_len', 
                                 'Min_axis_len', 'Eccentricity', 'Solidity']
        
    mask_bin_name = img_name[:-4]+'_bin_mask.tif'
    mask_lab_name = img_name[:-4]+'_labeled_mask.tif'
    mask_prop_name = img_name[:-4]+'_mask_props.csv'
    
    mask_bin = mask_binary.astype(np.uint16)
    mask_lab = mask_labeled.astype(np.uint16)
    
    imsave(mask_bin_name, mask_bin)
    imsave(mask_lab_name, mask_lab)
    mask_props_df.to_csv(mask_prop_name, index = None)
    
    return mask_binary, mask_labeled

def combine_props(DL_path):
    props_list = sorted(getListFiles(DL_path, 'props.csv'))
    if len(props_list) == 0:
        print('No props file found. Please generate props file!')
    
    combined = []
    for k in range(len(props_list)):
        fname = props_list[k].split('\\')[-1].split('_')[0]
        data = pd.read_csv(props_list[k])
        data['Filename'] = fname
        data['Num_obj'] = len(data)
        data = data[['Filename', 'Num_obj', 'ID', 'Centroid_X', 'Centroid_Y',
                     'Area', 'Perimeter', 'Circularity', 'Maj_axis_len', 
                     'Min_axis_len', 'Eccentricity', 'Solidity']]
        combined.append(data)
    combined_df = pd.concat(combined)
    combined_arr = np.asarray(combined_df)
    possible_empty_ind = np.where(combined_df['Num_obj']==1)[0]
    #empty_ind = []
    for ind in possible_empty_ind:
        if combined_arr[ind, 5] == 0 and combined_arr[ind, 6] == 0:
            combined_arr[ind, 1] = 0
    combined_df = pd.DataFrame(combined_arr)
    combined_df.columns = ['Filename', 'Num_obj', 'ID', 'Centroid_X', 'Centroid_Y',
                 'Area', 'Perimeter', 'Circularity', 'Maj_axis_len', 
                 'Min_axis_len', 'Eccentricity', 'Solidity']
    combined_df.to_csv(DL_path+'\\'+'Combined_props.csv', index = None)

DL_list = sorted(getListFiles(DL_path, '.tif'))
for img_name in DL_list:
    mask_binary, mask_labeled = get_mask(img_name, size_upper, size_lower, px_size, SNR_thres)

combine_props(DL_path)