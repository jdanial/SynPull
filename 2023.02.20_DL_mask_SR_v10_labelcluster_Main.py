# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:19:21 2022

@author: Aloe Vera
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
import astroalign as aa

DL_path = r'E:\Synaptosomes_270124\2024.01.31_0.7 to 2.2\Green_converted'
SR_path = r'E:\Synaptosomes_270124\2024.01.31_0.7 to 2.2\raw'

size_upper = 2200 # nm, maximum diameter of the dot
size_lower = 700 # nm, minimum diameter of the dot

px_size = 105 # pixel size of the image

mag = 10 # magnification when reconstructing SR image

SNR_thres = 2

apply_AffineTransform = True # Change to false if you don't want to apply affine transform
source_channel = 'Green' # Change to 'Blue' if use the 488 channel
target_channel = 'Red' # target channel

# Affine transform
im_path = r'E:\Synaptosomes_270124\affine_transform'
Blue_path = im_path + '\\bluebeads'
Red_path = im_path + '\\redbeads'
Green_path = im_path + '\\greenbeads'
load_previous = True

def getListFiles(path, kwd = ''):
    filelist = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            if kwd in filespath:
                filelist.append(os.path.join(root,filespath)) 
    return filelist

def getListFiles2(path, kwd1 = '', kwd2 = ''):
    filelist = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            if kwd1 in filespath and kwd2 in filespath :
                filelist.append(os.path.join(root,filespath)) 
    return filelist

def getListFiles_exclude(path, kwd_in = '', kwd_ex = ''):
    filelist = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            if kwd_in in filespath and kwd_ex not in filespath :
                filelist.append(os.path.join(root,filespath)) 
    return filelist

def get_previous_affmx(im_path):
    affmx_names = getListFiles(im_path, 'affmx.csv')
    blue_red_H = []
    green_red_H = []
    for mx in affmx_names:
        if 'Blue_red' in mx:
            blue_red_H = np.asarray(pd.read_csv(mx, header = None))
        if 'Green_red' in mx:
            green_red_H = np.asarray(pd.read_csv(mx, header = None))
    regenerate_affmx = False
    if len(blue_red_H) == 0:
        print('Warning: Blue_to_red matrix not found. Regenerating...')
        regenerate_affmx = True
    if len(green_red_H) == 0:
        print('Warning: Green_to_red matrix not found. Regenerating...')
        regenerate_affmx = True
    return blue_red_H, green_red_H, regenerate_affmx

def avg_img(imgpath):
    imlist = getListFiles(imgpath)
    im_all = []
    for img_name in imlist:
        im = io.imread(img_name)
        im_all.append(im)
    im_stack = np.asarray(im_all)
    im_avg = np.mean(im_stack, axis = 0)
    im_avg_max = np.max(im_avg)
    im_avg_nor = np.round(im_avg/im_avg_max*255)
    im_avg_nor = im_avg_nor.astype(np.uint8)
    
    return im_avg_nor

def aff_trans(im1_name, im2_name):
    im1 = Image.open(im1_name)
    im2 = Image.open(im2_name)
    registered_image, footprint = aa.register(im1, im2)
    transf, (source_list, target_list) = aa.find_transform(im1, im2)
    return registered_image, transf.params

def add_clustered(DL_list, sorted_SR_list, DL_path, SR_path, DL_names):
    if len(DL_list) == len(sorted_SR_list):
        return sorted_SR_list
    else:
        print('Missing clustered file, generating...')
        missing_file = []
        for file in DL_list:
            imname = file.split('\\')[-1][:-4]
            exist = 0
            for imname1 in sorted_SR_list:
                if imname in imname1:
                    exist = 1
                    break
            if exist == 0:
                missing_file.append(file)
        for missing in missing_file:
            missing_csv = missing.replace('.tif', '_clustered_75.0_2.csv')
            missing_csv = missing_csv.replace(DL_path, SR_path)
            cols = ['', 'Unnamed: 0', 'id', 'frame', 'x [nm]', 
                             'y [nm]', 'sigma [nm]', 'intensity [photon]',
                             'offset [photon]', 'bkgstd [photon]', 'chi2',
                             'uncertainty_xy [nm]', 'X', 'Y', 'DBSCAN_label']
            zero_row = pd.DataFrame(np.zeros((1,len(cols))))
            zero_row.columns = cols
            zero_row.to_csv(missing_csv, index = None)
        SR_csv_list = getListFiles(SR_path, 'clustered')
        sorted_SR_list = sort_csv(DL_names, SR_csv_list)
    return sorted_SR_list
        
    
def add_cluster_profile(sorted_SR_list, sorted_cluster_list, DL_names, SR_path):
    if len(sorted_SR_list) == len(sorted_cluster_list):
        pass
    else:
        print('Missing clusterProfile file, generating...')
        missing_file = []
        for file in sorted_SR_list:
            imname = file.split('\\')[-1].split('_')[0]
            exist = 0
            for imname1 in sorted_cluster_list:
                if imname in imname1:
                    exist = 1
                    break
            if exist == 0:
                missing_file.append(file)    
        for missing in missing_file:
            data = pd.read_csv(missing)
            zero_row = pd.DataFrame(np.zeros(data.shape[1])).T
            zero_row.columns = data.columns
            data = pd.concat((data, zero_row), axis = 0)
            data.to_csv(missing)
            clustprof = missing.replace('clustered', 'clusterProfile')
            clustprof_col = ['', 'cluster_id', 'area', 'X_(px)', 'Y_(px)', 
                             'convex_area', 'major_axis_length', 'minor_axis_length',
                             'eccentricity', 'xMin', 'yMin', 'xMax', 'yMax', 
                             'length', 'n_localisation']
            clust_zeros = np.zeros((1, len(clustprof_col)))
            clust_pd = pd.DataFrame(clust_zeros, columns = clustprof_col, index = None)
            clust_pd.to_csv(clustprof, index = None)
        
        SR_csv_list = getListFiles(SR_path, 'clustered')
        SR_cluster_list = getListFiles_exclude(SR_path, 'clusterProfile', 'corrected')
        sorted_SR_list = sort_csv(DL_names, SR_csv_list)
        sorted_cluster_list = sort_csv(DL_names, SR_cluster_list)
    return sorted_SR_list, sorted_cluster_list
        
if load_previous:
    blue_red_H, green_red_H, regenerate_affmx = get_previous_affmx(im_path)
    if regenerate_affmx:
        im_blue = avg_img(Blue_path)
        im_red = avg_img(Red_path)
        im_green =  avg_img(Green_path)
        blue_name = im_path+'\\Blue_avg.tif'
        red_name = im_path+'\\Red_avg.tif'
        green_name = im_path+'\\Green_avg.tif'
        imsave(blue_name, im_blue)
        imsave(red_name, im_red)
        imsave(green_name, im_green)
        
        aligned_blue, blue_red_H = aff_trans(blue_name, red_name)
        aligned_green, green_red_H = aff_trans(green_name, red_name)
        
        blue_red_mx = pd.DataFrame(blue_red_H)
        blue_red_mx.to_csv(im_path+'\\Blue_red_affmx.csv', header = None, index = None)
        imsave(im_path+'\\Aligned_blue_to_red.tif', aligned_blue.astype(np.uint8))
        green_red_mx = pd.DataFrame(green_red_H)
        green_red_mx.to_csv(im_path+'\\Green_red_affmx.csv', header = None, index = None)
        imsave(im_path+'\\Aligned_green_to_red.tif', aligned_green.astype(np.uint8))
else:
    im_blue = avg_img(Blue_path)
    im_red = avg_img(Red_path)
    im_green =  avg_img(Green_path)
    blue_name = im_path+'\\Blue_avg.tif'
    red_name = im_path+'\\Red_avg.tif'
    green_name = im_path+'\\Green_avg.tif'
    imsave(blue_name, im_blue)
    imsave(red_name, im_red)
    imsave(green_name, im_green)
    
    aligned_blue, blue_red_H = aff_trans(blue_name, red_name)
    aligned_green, green_red_H = aff_trans(green_name, red_name)
    
    blue_red_mx = pd.DataFrame(blue_red_H)
    blue_red_mx.to_csv(im_path+'\\Blue_red_affmx.csv', header = None, index = None)
    imsave(im_path+'\\Aligned_blue_to_red.tif', aligned_blue.astype(np.uint8))
    green_red_mx = pd.DataFrame(green_red_H)
    green_red_mx.to_csv(im_path+'\\Green_red_affmx.csv', header = None, index = None)
    imsave(im_path+'\\Aligned_green_to_red.tif', aligned_green.astype(np.uint8))



#%%
def get_mask(img_name, size_upper, size_lower, px_size, SNR_thres, 
             apply_AffineTransform, source_channel, target_channel,
             blue_red_H, green_red_H):
    
    im = io.imread(img_name)
    if len(im.shape)==3:
        im = np.mean(im, axis = 0)
    if apply_AffineTransform == True:
        if source_channel == 'Blue':
            im = cv2.warpAffine(im, blue_red_H[0:2,:], [im.shape[1], im.shape[0]])
        elif source_channel == 'Green':
            im = cv2.warpAffine(im, green_red_H[0:2,:], [im.shape[1], im.shape[0]])
    
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
    
    
def sort_csv(DL_names, SR_csv_list):
    sorted_SR_list = []
    for name in DL_names:
        for SR_name in SR_csv_list:
            if name in SR_name:
                sorted_SR_list.append(SR_name)
    return sorted_SR_list

def correct_cluster(SR_file, cluster_file):
    corrected_clustername = cluster_file[:-4]+'_corrected.csv'
    SR_data = pd.read_csv(SR_file)
    if len(SR_data) == 0:
        SR_data.loc[len(SR_data)] = list(np.zeros(len(SR_data.columns)))
    cluster_ID_list = list(range(int(1+np.max(SR_data['DBSCAN_label']))))
    xy = []
    for ID in cluster_ID_list:
        x_mean = np.mean(SR_data[SR_data['DBSCAN_label'] == ID]['X'])
        y_mean = np.mean(SR_data[SR_data['DBSCAN_label'] == ID]['Y'])
        xy.append([x_mean, y_mean])
    XY = pd.DataFrame(xy)
    XY.columns = ['X_DL', 'Y_DL']
    cluster_data = pd.read_csv(cluster_file)
    cluster_data['X_DL'] = XY['X_DL']
    cluster_data['Y_DL'] = XY['Y_DL']
    cluster_data.to_csv(corrected_clustername, index = None)
    return corrected_clustername
    
def label_csv(mask_labeled, SR_csv):
    SR_data = pd.read_csv(SR_csv)
    loc_xy = np.asarray(round(SR_data[['X_DL', 'Y_DL']])) # Read position of centroid
    #loc_px = np.asarray(round(loc_xy/px_size))
    labels = []
    for i in range(len(loc_xy)):
        x = int(loc_xy[i][1])
        y = int(loc_xy[i][0])
        label = mask_labeled[x, y]
        labels.append(label-1)
    SR_data['Synaptosome_ID'] = labels
    SR_data.to_csv(SR_csv[:-4]+'_synID.csv', index = None)
    
    SR_data_clear = SR_data[SR_data.Synaptosome_ID != -1]
    SR_data_clear.to_csv(SR_csv[:-4]+'_synID_cleared.csv', index = None)
    return SR_data, SR_data_clear

def combine_SR_cleared(SR_path):
    cleared_list = sorted(getListFiles(SR_path, 'cleared.csv'))
    combine_SR = []
    for i in range(len(cleared_list)):
        data = pd.read_csv(cleared_list[i])
        data['Filename'] = cleared_list[i].split('\\')[-1].split('_')[0]
        data['Cluster_num'] = len(data)
        col_list = ['Filename', 'Cluster_num']+list(data.columns)
        col_list = col_list[:-2]
        col_list1 = []
        for col in col_list:
            if 'Unnamed' not in col:
                col_list1.append(col)
        data = data[col_list1]
        combine_SR.append(data)
    combine_df = pd.concat(combine_SR)
    combine_df.to_csv(SR_path+'\\Combined_SR_clear.csv', index = None)

def combine_SR_negative(SR_path):
    alldata_list = sorted(getListFiles(SR_path, 'synID.csv'))
    neg_data = []
    for d in alldata_list:
        data = pd.read_csv(d)
        data_neg = data[data.Synaptosome_ID == -1]
        neg_data.append(data_neg)
    combine_SR_neg = []
    for i in range(len(neg_data)):
        data = neg_data[i]
        data['Filename'] = alldata_list[i].split('\\')[-1].split('_')[0]
        data['Cluster_num'] = len(data)
        col_list = ['Filename', 'Cluster_num']+list(data.columns)
        col_list = col_list[:-2]
        col_list1 = []
        for col in col_list:
            if 'Unnamed' not in col:
                col_list1.append(col)
        data = data[col_list1]
        combine_SR_neg.append(data)
    combine_neg = pd.concat(combine_SR_neg)
    combine_neg.to_csv(SR_path+'\\Combined_SR_negative.csv', index = None)

def getinfo_DL_SR(DL_path, SR_path):
    DL_info_fname = DL_path +'\\Combined_props.csv'
    DL_info = pd.read_csv(DL_info_fname)
    DL_uinfo = list(pd.unique(DL_info['Filename']))
    DL_num_obj = []
    SR_in = []
    SR_out = []
    for i in range(len(DL_uinfo)):
        obj_ind = np.where(DL_info['Filename'] == DL_uinfo[i])[0]
        DL_num_obj.append(DL_info['Num_obj'][obj_ind[0]])
    for DL_name in DL_uinfo:
        SR_info_fname = getListFiles2(SR_path, DL_name, 'synID.csv')[0]
        SR_info = pd.read_csv(SR_info_fname)
        num_out = len(SR_info['Synaptosome_ID'][SR_info['Synaptosome_ID'] == -1])
        num_in = len(SR_info['Synaptosome_ID']) - num_out
        SR_in.append(num_in)
        SR_out.append(num_out)
    DL_SR_info = pd.DataFrame([DL_uinfo, DL_num_obj, SR_in, SR_out]).T
    DL_SR_info.columns = ['Filename', 'Num_of_obj', 'Num_clus_in', 'Num_clus_out']
    DL_SR_info.to_csv(SR_path+'\\DL_SR_cluster_info.csv', index = None)
    
    
DL_list = sorted(getListFiles(DL_path, '.tif'))
DL_names = [f.split('\\')[-1][:-4] for f in DL_list]
SR_csv_list = getListFiles(SR_path, 'clustered')
SR_cluster_list = getListFiles_exclude(SR_path, 'clusterProfile', 'corrected')
sorted_SR_list = sort_csv(DL_names, SR_csv_list)
sorted_cluster_list = sort_csv(DL_names, SR_cluster_list)

sorted_SR_list = add_clustered(DL_list, sorted_SR_list, DL_path, SR_path, DL_names)
sorted_SR_list, sorted_cluster_list = add_cluster_profile(sorted_SR_list, 
                                                          sorted_cluster_list, DL_names, SR_path)

for i in range(0, len(DL_names)):
    img_name = DL_list[i]
    SR_csv = sorted_SR_list[i]
    SR_cluster = sorted_cluster_list[i]
    mask_binary, mask_labeled =  get_mask(img_name, size_upper, size_lower, px_size, SNR_thres, 
             apply_AffineTransform, source_channel, target_channel,
             blue_red_H, green_red_H)
    corrected_clustername = correct_cluster(SR_csv, SR_cluster)
    SR_data, SR_data_clear = label_csv(mask_labeled, corrected_clustername)

combine_props(DL_path)
combine_SR_cleared(SR_path)
getinfo_DL_SR(DL_path, SR_path)
combine_SR_negative(SR_path)