import pickle
import struct
import time
import queue
import time
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import yaml
import argparse
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import pandas as pd
from .utils import get_CSI, preprocess_CSI, get_dfs, get_csi_dfs
import torch.nn.functional as F


Bone_Connections = [
    (0,1),(1,2),(2,3),  # head and torso
    (4,5), (5,6), (6,7), # right arm
    (8,9), (9,10), (10,11), # left arm
    (12,13), (13,14), (14,15), # right leg
    (16,17), (17,18), (18,19), # left leg
    (0,16), (0,12), 
]

Activities = ['drawcircleclockwise', 'liftupahand', 'pushhandaway', 'shakehead', 'drawtriangle',
              'pushup', 'playphone', 'turn', 'sweep', 'answerphone', 'pullhandin', 'walk', 'sit',
              'yawn', 'sleep', 'spreadandpinch', 'drawzigzag', 'handwave', 'clap', 'lunge', 
              'drawcirclecounterclockwise', 'squat', 'eat', 'slide', 'stopsign', 'pickup', 
              'legraise', 'tap', 'falldown', 'dance', 'drink', 'type']
# spreadandpinch, eat, yaw, and playphone have the least amount of data

user_ids = [1,2,3,4,5,6,7,8,9,10]  # user 7 has the least amount of data


class OctonetMini(Dataset):
    def __init__(self, config):
        self.config = config
        self.task = config['task']           # the target task: activity_recognition, pose_estimation
        self.pose_time_length = config['pose_time_length']  # the time length of the csi or dfs data for pose estimation, only used when the task is pose_estimation
        self.dataset_path = config['dataset_path']
        self.user_selected = config["user_list"]  # the selected user list:: [1,2,3,4,5,6,7,8,9,10]
        self.activity_slected = config["activity_list"]  # the selected activity: tuple (start activity_id, end activity_id) :: 32 activities in total
        self.format = config['format']       # the format of the data: 'polar', 'cartesian', 'complex', 'dense_dfs', 'dense_dfs_amp'
        self.tqdm_disable = config['tqdm_disable']
        self.time_length = config['time_length']
        
        self.data, self.activity, self.user_id, self.skeleton_data = self._load_data()
        
        if self.activity_slected[1] > len(Activities):
            raise ValueError("The activity_slected[1] is out of range.")
        else:
            self.activity_types = Activities[self.activity_slected[0]:self.activity_slected[1]]
        self.activity2actID = {act: i for i, act in enumerate(self.activity_types)}
        # get the index of the valid samples that have the activity in self.activity_types
        self.valid_activtiy_index = [i for i, act in enumerate(self.activity) if act in self.activity_types]
        self.valid_user_index = [i for i, user in enumerate(self.user_id) if user in self.user_selected]
        self.valid_index = list(set(self.valid_activtiy_index) & set(self.valid_user_index))
        
        # Valid data based on the selected user and activity
        self.wifi_data = [self.data[i] for i in self.valid_index]
        self.activity_names = [self.activity[i] for i in self.valid_index]
        self.user_ids = [self.user_id[i] for i in self.valid_index]
        self.skeleton_data_list = [self.skeleton_data[i] for i in self.valid_index]
        
        if self.task == 'pose_estimation':
            # the label will be the last frame of the skeleton data
            self.label = [skeleton_data[-1] for skeleton_data in self.skeleton_data_list]
        elif self.task == 'activity_recognition':
            self.label = [self.activity2actID[act] for act in self.activity_names]
        else:
            raise ValueError("The task is not supported")
        
    def _load_data(self):
        data_list = []
        activity_lable_list = []
        user_id_list = []
        skeleton_data_list = []
        
        if self.format == 'dense_dfs' or self.format == 'dense_dfs_amp':
            dfs_file_folder = self.dataset_path + 'DFS/'
            dfs_file_names = [name for name in os.listdir(dfs_file_folder) if name.endswith('.npy')]
            for name in tqdm(dfs_file_names, disable=self.tqdm_disable):
                sample = np.load(dfs_file_folder + name) # shape [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
                if sample.shape[0] <= 5:  # filer out the sample with Time_bins < 5 e.g.，time recording less than 5*0.25s = 1.25s
                    continue
                
                if self.task == 'pose_estimation': # then, we will sperate the data along the time axis into multiple samples to make the pose estimation diverse
                    if self.format == 'dense_dfs_amp':
                        sample = np.sqrt(sample[...,0]**2 + sample[...,1]**2)
                        sample = np.expand_dims(sample, axis=-1)
                    skeleton_data = np.load(self.dataset_path + 'Skeleton/' + name.replace('dfs', 'skeleton'))    
                    skeleton_time_length = skeleton_data.shape[0]    
                    time_scale_skeleton_to_dfs = skeleton_time_length // sample.shape[0]
                    slice_length = self.pose_time_length // 20
                    stride = slice_length // 2
                    sample_time_dim = sample.shape[0]
                    skeleton_time_dim = skeleton_data.shape[0]
                    for i in range(0, sample_time_dim-slice_length+1, stride):
                        data_list.append(sample[i:i+slice_length])
                        activity_lable_list.append(name.split('_')[2])
                        user_id_list.append(int(name.split('_')[1]))
                        skeleton_data_list.append(skeleton_data[i*time_scale_skeleton_to_dfs:(i+slice_length)*time_scale_skeleton_to_dfs])
                elif self.task == 'activity_recognition':
                    skeleton_data = np.load(self.dataset_path + 'Skeleton/' + name.replace('dfs', 'skeleton'))
                    if sample.shape[0] > self.time_length // 20:  # cut the sample to the time_length, 20 is based on the stft stride
                        sample = sample[:self.time_length//20]
                    else:
                        sample = np.pad(sample, ((0,self.time_length//20 -sample.shape[0]), (0,0), (0,0), (0,0), (0,0)), 'constant', constant_values=0)
                    if self.format == 'dense_dfs_amp':
                        sample = np.sqrt(sample[...,0]**2 + sample[...,1]**2)
                        sample = np.expand_dims(sample, axis=-1)
                    data_list.append(sample)
                    activity_lable_list.append(name.split('_')[2])
                    user_id_list.append(int(name.split('_')[1]))       
                    skeleton_data_list.append(skeleton_data)
                else:
                    raise ValueError("The task is not supported")
        else:
            csi_folder = self.dataset_path + 'CSI_Processed/'
            csi_file_names = [name for name in os.listdir(csi_folder) if name.endswith('.npy')] 
            for name in tqdm(csi_file_names, disable=self.tqdm_disable):
                sample = np.load(csi_folder + name) # shape [Time_length, num_subcarriers, Rx*Tx, 1] complex64
                sample = np.squeeze(sample, axis=-1) # shape [Time_length, num_subcarriers, Rx*Tx]
                if sample.shape[0] <= 5 * 20:
                    continue
                
                if self.task == 'pose_estimation': # then, we will sperate the data along the time axis into multiple samples to make the pose estimation diverse
                    skeleton_data = np.load(self.dataset_path + 'Skeleton/' + name.replace('csi', 'skeleton'))
                    skeleton_time_length = skeleton_data.shape[0]    
                    slice_length = self.pose_time_length
                    stride = slice_length // 2
                    data_time_dim = sample.shape[0]
                    for i in range(0, data_time_dim-slice_length+1, stride):
                        activity_lable_list.append(name.split('_')[2])
                        user_id_list.append(int(name.split('_')[1]))
                        skeleton_data_list.append(skeleton_data[i:i+slice_length])
                        temp = sample[i:i+slice_length]
                        if self.format == 'polar':
                            temp_amp = np.abs(temp)
                            temp_phase = np.angle(temp)
                            temp = np.concatenate((temp_amp, temp_phase), axis=-1) # shape: [Time_length, num_subcarriers, Rx*Tx*2]
                        elif self.format == 'cartesian':
                            temp_real = np.real(temp)
                            temp_imag = np.imag(temp)
                            temp = np.concatenate((temp_real, temp_imag), axis=-1) # shape: [Time_length, num_subcarriers, Rx*Tx*2]
                        elif self.format == 'amplitude':
                            temp = np.abs(temp)  # shape: [Time_length, num_subcarriers, Rx*Tx]
                        elif self.format == 'complex': # shape: [Time_length, num_subcarriers, Rx*Tx] in complex64
                            pass
                        else:
                            raise ValueError("The format is not supported")
                        data_list.append(temp)
                elif self.task == 'activity_recognition':
                    skeleton_data = np.load(self.dataset_path + 'Skeleton/' + name.replace('csi', 'skeleton'))
                    if sample.shape[0] > self.time_length:  # cut the sample to the time_length, 20 is based on the stft stride
                        sample = sample[:self.time_length]
                    else:
                        sample = np.pad(sample, ((0, self.time_length -sample.shape[0]), (0,0), (0,0)), 'constant', constant_values=0)
                    if self.format == 'polar':
                        sample_amp = np.abs(sample)
                        sample_phase = np.angle(sample)
                        sample = np.concatenate((sample_amp, sample_phase), axis=-1) # shape: [Time_length, num_subcarriers, Rx*Tx*2]
                    elif self.format == 'cartesian':
                        sample_real = np.real(sample)
                        sample_imag = np.imag(sample)
                        sample = np.concatenate((sample_real, sample_imag), axis=-1) # shape: [Time_length, num_subcarriers, Rx*Tx*2]
                    elif self.format == 'amplitude':
                        sample = np.abs(sample)    # shape: [Time_length, num_subcarriers, Rx*Tx]
                    elif self.format == 'complex': # shape: [Time_length, num_subcarriers, Rx*Tx] in complex64
                        pass
                    data_list.append(sample)
                    activity_lable_list.append(name.split('_')[2])
                    user_id_list.append(int(name.split('_')[1]))       
                    skeleton_data_list.append(skeleton_data)
                else:
                    raise ValueError("The task is not supported")
        return data_list, activity_lable_list, user_id_list, skeleton_data_list
        

    def __getitem__(self, index):
        return torch.tensor(self.wifi_data[index]), torch.tensor(self.label[index]), self.activity_names[index], self.user_ids[index], self.skeleton_data_list[index][-1]
            
    def __len__(self):
        return len(self.wifi_data)
        

# converting the data shape to the shape that the model needs
class OctonetMini_data_shape_converter:
    def __init__(self,config):
        self.model_input_shape = config['model_input_shape'] # BCHW, BLC, BTCHW, B2CNFT 
        # B: batch size, C: channel, H: height, W: width, T: time length, N: number of (WiFi CSI) links, F: frequency bins
        self.format = config['format']
        try:
            self.z_score = config['z_score']
        except:
            self.z_score = False
        try:
            self.ppe = config['ppe'] # the flag of the physics prior embedding, None: no ppe, otherwise: the ppe is used
        except:
            self.ppe = None
        # viral phase with zero phase
        self.dense_dfs_virtual_phase = np.zeros(61)
        # viral phase with zero phase
        self.csi_virtual_phase = np.zeros(114)
        self.config = config
        
    def shape_convert(self,batch):
        if self.format == 'dense_dfs' or self.format == 'dense_dfs_amp':    # shape of the input batch: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2 or 1]
            try:
                self.align2widar3 = self.config['align2widar3']
            except:
                # print('no align2widar3')
                self.align2widar3 = False
            if self.align2widar3: # align the input batch to the widar3 dataset: [batch_size, Time_bins=256, freq_bins=121, num_subcarriers=30, Rx*Tx=3, 2or1]
                # Efficiently align OctoNetMini data dimensions to Widar3 data shape
                # Create a new tensor with Widar3 dimensions
                widar3_shape = (batch.shape[0], 256, 121, 30, 3, batch.shape[5])
                aligned_batch = torch.zeros(widar3_shape, dtype=batch.dtype, device=batch.device)
                
                # Time dimension: interpolate from 60 to 256 using F.interpolate
                # First reshape to make time the last dimension for interpolation
                reshaped_batch = batch.permute(0, 2, 3, 4, 5, 1)  # [B, F, S, R, C, T]
                batch_size, freq_bins, subcarriers, rx_tx, channels, time_bins = reshaped_batch.shape
                
                # Flatten all dimensions except time for interpolation
                flat_batch = reshaped_batch.reshape(-1, time_bins)
                # Interpolate time dimension
                interpolated_time = F.interpolate(
                    flat_batch.unsqueeze(1),  # Add channel dim for interpolate
                    size=256,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)  # Remove channel dim
                # Reshape back with new time dimension
                time_aligned = interpolated_time.reshape(batch_size, freq_bins, subcarriers, rx_tx, channels, 256)
                # Permute back to original dimension order
                time_aligned = time_aligned.permute(0, 5, 1, 2, 3, 4)  # [B, T, F, S, R, C]  
                # Frequency dimension: interpolate from 61 to 121
                freq_reshaped = time_aligned.permute(0, 1, 3, 4, 5, 2)  # [B, T, S, R, C, F]
                flat_freq = freq_reshaped.reshape(-1, freq_bins)
                interpolated_freq = F.interpolate(
                    flat_freq.unsqueeze(1),
                    size=121,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
                freq_aligned = interpolated_freq.reshape(batch_size, 256, subcarriers, rx_tx, channels, 121)
                freq_aligned = freq_aligned.permute(0, 1, 5, 2, 3, 4)  # [B, T, F, S, R, C]
                # Subcarriers: downsample from 114 to 30 (take every ~4th subcarrier)
                subcarrier_indices = torch.linspace(0, subcarriers-1, 30).long()
                subcarrier_aligned = freq_aligned[:, :, :, subcarrier_indices, :, :]
                # Rx*Tx: take first 3 of 4
                rx_tx_aligned = subcarrier_aligned[:, :, :, :, :3, :]
                # Replace the original batch with the aligned one
                batch = rx_tx_aligned
                
            if self.z_score:
                # the input batch is the torch.tensor dfs data with shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
                # z score normalization except the the first dimension (batch_size) and the last dimension (real and imag)
                mean = batch.mean(dim=[1,2,3,4], keepdim=True)
                std = batch.std(dim=[1,2,3,4], keepdim=True)
                batch = (batch - mean) / std
            if self.model_input_shape == 'BCHW':
                # merge the last three dimensions of the batch into one dimension: num_subcarriers * Rx*Tx * 2
                batch = batch.reshape(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]*batch.shape[4]*batch.shape[5]) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers * Rx*Tx * 2]
                # transpose the last dim to the second dim as channel
                batch = batch.permute(0,3,1,2)
            elif self.model_input_shape == 'BCHW-C': # the complex version of the BCHW
                if  self.format == 'dense_dfs_amp' and self.ppe is None:
                    # the batch now is the amplitude of the dfs data with shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx*(num_receivers), 1]
                    # convert to tensor and repeat to have the same shape as the amplitude
                    virtual_phase = torch.tensor(self.dense_dfs_virtual_phase).reshape(1, 1, self.dense_dfs_virtual_phase.shape[0], 1, 1, 1).repeat(batch.shape[0], batch.shape[1], 1, batch.shape[3], batch.shape[4], batch.shape[5])
                    real_part = batch*torch.cos(virtual_phase)
                    img_part = batch*torch.sin(virtual_phase)
                else:
                    real_part = batch[:,:,:,:,:,0]
                    img_part = batch[:,:,:,:,:,1]
                # convert to the complex.64
                batch = real_part + 1j*img_part # shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx]
                # merge the last two dimensions of the batch into one dimension: num_subcarriers*Rx*Tx
                batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]*batch.shape[4]) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers * Rx*Tx]
                # transpose the last dim to the second dim as channel
                batch = batch.permute(0,3,1,2)  # shape: [batch_size, num_subcarriers*Rx*Tx, Time_bins, freq_bins]
            elif self.model_input_shape == 'BLC':
                # merge the last four dimensions of the batch into one dimension: freq_bins*num_subcarriers * Rx*Tx * 2
                batch = batch.reshape(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]*batch.shape[4]*batch.shape[5]) # shape: [batch_size, Time_bins, freq_bins*num_subcarriers * Rx*Tx * 2]
            elif self.model_input_shape == 'BTCHW': # for baseline: Widar3.0 model
                # merge the last two dimensions of the batch into one dimension: Rx*Tx * 2
                batch = batch.reshape(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]*batch.shape[5]) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx * 2]
                # transpose the last dim to the third dim as channel
                batch = batch.permute(0,1,4,2,3)  # shape: [batch_size, Time_bins, Rx*Tx * 2, freq_bins, num_subcarriers]
            elif self.model_input_shape == 'B2CNFT':  # for baseline: SLNet
                if self.format == 'dense_dfs_amp' and self.ppe is None:
                    # the batch now is the amplitude of the dfs data with shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx*(num_receivers), 1]
                    # convert to tensor and repeat to have the same shape as the amplitude
                    virtual_phase = torch.tensor(self.dense_dfs_virtual_phase).reshape(1, 1, self.dense_dfs_virtual_phase.shape[0], 1, 1, 1).repeat(batch.shape[0], batch.shape[1], 1, batch.shape[3], batch.shape[4], batch.shape[5])
                    real_part = batch*torch.cos(virtual_phase)
                    img_part = batch*torch.sin(virtual_phase)
                    batch = torch.cat((real_part, img_part), dim=-1) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx*(num_receivers), 2]
                batch = batch.permute(0,5,3,4,2,1)  # shape: [batch_size, 2, num_subcarriers, Rx*Tx, freq_bins, Time_bins]
        elif self.format == 'polar' or self.format == 'cartesian' or self.format == 'amplitude': 
            # shape of the input batch: [batch_size, frames, subcarriers, Rx*Tx*2] or [batch_size, frames, subcarriers, Rx*Tx]
            if self.model_input_shape == 'BCHW':
                batch = batch.permute(0,3,1,2) # shape: [batch_size, Rx*Tx*2, frames, subcarriers]
            elif self.model_input_shape == 'BLC':
                batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]) # shape: [batch_size, frames, subcarriers * Rx*Tx * 2]
            elif self.model_input_shape == 'BTCHW' or self.model_input_shape == 'B2CNFT':
                raise ValueError("The config.model_input_shape 'BTCHW' or 'B2CNFT'  only supported by config.format 'dfs'.")
            else:
                raise ValueError("The config.model_input_shape must be one of 'BCHW', 'BLC'.")
        elif self.format == 'complex':
            # shape of the input batch: [batch_size, frames, subcarriers, Rx*Tx]  in torch.complex64
            csi_amp = torch.abs(batch)
            virtual_phase = torch.tensor(self.csi_virtual_phase).view(1, 1, self.csi_virtual_phase.shape[0], 1).repeat(batch.shape[0], batch.shape[1], 1,  batch.shape[3])
            batch = csi_amp*torch.cos(virtual_phase) + 1j*csi_amp*torch.sin(virtual_phase)
            if self.model_input_shape == 'BCHW-C':
                batch = batch.permute(0,3,1,2) # shape: [batch_size, Rx*Tx, frames, subcarriers]
            # elif self.model_input_shape == 'BLC':
            #     batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]) # shape: [batch_size, frames, subcarriers * Rx*Tx]
            elif self.model_input_shape == 'BTCHW' or self.model_input_shape == 'B2CNFT':
                raise ValueError("The config.model_input_shape 'BTCHW' or 'B2CNFT'  only supported by config.format 'dfs'.")
            else:
                raise ValueError("The config.model_input_shape must be one of 'BCHW-C'.")
        else:
            raise ValueError("The config.format must be one of 'polar', 'cartesian', 'dfs', 'complex'.")
        return batch

def make_OctonetMini_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd, num_workers = 0):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_padd,
        shuffle=is_training,
        drop_last=True,
        generator=generator,
        num_workers=num_workers,
    )
    return loader

