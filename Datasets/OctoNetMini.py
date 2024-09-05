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

### some utility functions for CSI data
def read_csi_all(file_path):
    csi_all = []
    ts_all = []
    header_all = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                # Read time
                ts_b = f.read(19).decode('utf-8')
                time = int(ts_b)
                ts_all.append(time)
                # Read header length, data_num, and data_len
                hdr_len, data_num, data_len = struct.unpack('>III', f.read(12))
                # Read header
                header = f.read(hdr_len)
                header_all.append(header)
                # Read CSI data
                csi_data = []
                for _ in range(data_num):
                    data = f.read(int(data_len / data_num))
                    csi_data.append(data)
                csi_all.append(csi_data)
            except Exception as e:
                break
    return csi_all, ts_all, header_all
    
    
def bytes_arr_to_complex_(v):
    v_complex = []
    for i in range(0, len(v), 4):
        try:
            R = v[i] if v[i+1]==0 else -(256-v[i])
            I = v[i+2] if v[i+3]==0 else -(256-v[i+2])
            comp = complex(R, I)
            v_complex.append(comp)
        except Exception as e:
            print("parse error, break")
            break
    return v_complex


def parse_header(header):
    def parse_from_range(header, start, end, dtype="int"):
        # print("start: ", start, "end: ", end)
        # print(header[start:end])
        if dtype == "int":
            return int.from_bytes(header[start:end], byteorder='little')
        elif dtype == "mac":
            b = header[start:end]
            mac_str = ':'.join(['{:02X}'.format(byte) for byte in b])
            return mac_str
        elif dtype == "hex2dec":
            return int.from_bytes(header[start:end], byteorder='little', signed=False)
    result = {}
    
    result["n_link"] = parse_from_range(header, 50, 51, dtype="hex2dec")
    result["n_subc"] = parse_from_range(header, 52, 56, dtype="hex2dec")
    result["rssi_a"] = parse_from_range(header, 60, 64, dtype="hex2dec")
    result["rssi_b"] = parse_from_range(header, 64, 68, dtype="hex2dec")
    result["mac_src"] = parse_from_range(header, 68, 74, dtype="mac")
    result["seq"] = parse_from_range(header, 76, 77, dtype="hex2dec")
    result['timestamp'] = parse_from_range(header, 88, 92, dtype="int")
    return result

def read_csi_data(file_path, csi_shape=(2,114), mac_filter='A4:A9:30:B1:AF:7D'):
    # (2, 114): two receive antennas and 114 subcarriers from the transmitter with mac address 'A4:A9:30:B1:AF:7D'
    # Only one antenna is activated for the transmitter
    csi_all, ts_all, header_all = read_csi_all(file_path)
    header_all_parsed = [parse_header(header) for header in header_all]
    # Convert CSI data to complex numbers for each record
    for i, csi_data in enumerate(csi_all):
        csi_data_complex = [bytes_arr_to_complex_(data) for data in csi_data]
        csi_all[i] = csi_data_complex
    
    csi_all_cleaned = []
    ts_all_cleaned = []
    header_all_cleaned = []
    for i, csi in enumerate(csi_all):
        csi = np.array(csi)
        # print(header_all_parsed[i])
        if (csi_shape and csi.shape != csi_shape):
            continue
        if (mac_filter and header_all_parsed[i]["mac_src"].upper() != mac_filter.upper()):
            continue
        csi_all_cleaned.append(csi)
        ts_all_cleaned.append(ts_all[i])
        header_all_cleaned.append(header_all_parsed[i])
    return csi_all_cleaned, ts_all_cleaned, header_all_cleaned


class OptitrackSkeleton():
    def __init__(self, optitrack_data_path):
        self.df_data = pd.read_csv(optitrack_data_path, skiprows=6)
        self.df_colum_label = pd.read_csv(optitrack_data_path, skiprows=2, nrows=1)
        self.df_colum_skeleton_label = pd.read_csv(optitrack_data_path, skiprows=3, nrows=1)
        self.bone_columns = [col for col in self.df_colum_label.columns if 'Bone' == col.split('.')[0]]
        self.bone_columns_index = [self.df_colum_label.columns.get_loc(col) for col in self.bone_columns]
        self.skeleton_labels = [self.df_colum_skeleton_label.columns[i] for i in self.bone_columns_index]
        self.landmarkers = list(set([label.split('.')[0] for label in self.skeleton_labels]))
        self.filtered_lamdmarkers = ['Skeleton-Aaron:RIndex2', 'Skeleton-Aaron:LPinky1', 'Skeleton-Aaron:LMiddle1', 'Skeleton-Aaron:LIndex3', 'Skeleton-Aaron:RPinky1',
                        'Skeleton-Aaron:LPinky3','Skeleton-Aaron:RMiddle2', 'Skeleton-Aaron:RRing1', 'Skeleton-Aaron:RIndex3', 'Skeleton-Aaron:LThumb3',
                        'Skeleton-Aaron:RRing3', 'Skeleton-Aaron:LRing1', 'Skeleton-Aaron:RThumb1', 'Skeleton-Aaron:Ab','Skeleton-Aaron:LMiddle3', 
                        'Skeleton-Aaron:LRing2', 'Skeleton-Aaron:RPinky2', 'Skeleton-Aaron:LThumb2', 'Skeleton-Aaron:RPinky3', 'Skeleton-Aaron:RThumb3',
                        'Skeleton-Aaron:LIndex1', 'Skeleton-Aaron:LIndex2', 'Skeleton-Aaron:LMiddle2', 'Skeleton-Aaron:RMiddle3', 'Skeleton-Aaron:RThumb2',
                        'Skeleton-Aaron:RMiddle1', 'Skeleton-Aaron:RRing2', 'Skeleton-Aaron:LPinky2', 'Skeleton-Aaron:LRing3', 'Skeleton-Aaron:LThumb1', 'Skeleton-Aaron:RIndex1']
        self.filtered_lamdmarkers_index = [i for i, label in enumerate(self.skeleton_labels) if label.split('.')[0] in self.filtered_lamdmarkers ]
        self.final_bone_columns_index = [self.bone_columns_index[i] for i in range(len(self.bone_columns_index)) if i not in self.filtered_lamdmarkers_index]
        self.final_skeleton_labels = [self.skeleton_labels[i] for i in range(len(self.skeleton_labels)) if i not in self.filtered_lamdmarkers_index]
        self.df_skeleton_data = self.df_data.iloc[:, self.final_bone_columns_index]
        self.time = self.df_data['Time (Seconds)'].values
        
    def get_skeleton_data(self, index):
        bone_pose = self.df_skeleton_data.iloc[index].values
        bone_pose = bone_pose.reshape((-1, 7))
        bone_positions = bone_pose[:, 4:7]
        bone_orientations = bone_pose[:, 0:4]
        return bone_positions, bone_orientations
    
    def get_skeleton_data_all(self):
        bone_positions_all = []
        bone_orientations_all = []
        for i in range(len(self.df_skeleton_data)):
            bone_pose = self.df_skeleton_data.iloc[i].values
            bone_pose = bone_pose.reshape((-1, 7))
            bone_positions = bone_pose[:, 4:7]
            bone_orientations = bone_pose[:, 0:4]
            bone_positions_all.append(bone_positions)
            bone_orientations_all.append(bone_orientations)
        return np.array(bone_positions_all), np.array(bone_orientations_all)
    
    def get_duration(self):
        return self.time[-1] - self.time[0]
    
    def get_sampling_rate(self):
        return len(self.time)/self.get_duration()
    
    def get_landmarkers(self):
        final_landmarkers = list(set([label.split('.')[0] for label in self.final_skeleton_labels]))
        return final_landmarkers

    def get_number_of_frames(self):
        return len(self.df_skeleton_data)
    
    
# Data loader for the OctoNetMini dataset
class multi_processing_data_loader:
    def __init__(self, config, data_path_list, label_path_list, cut_ratio_list, user_id_list, activity_id_list):
        self.preprocess = config['preprocess']
        self.format = config['format']
        self.time_length = config['time_length']
        self.num_classes = config['num_classes']   # this is the output vector length of the model, wich can be the number of activities, the number of the landmarkers of the human poses, or the length of the heart rate record
        self.tqdm_disable = config['tqdm_disable']
        
        self.task = config['task']
        self.modality = config['modality']
        self.data_path_list = data_path_list
        self.label_path_list = label_path_list
        self.cut_ratio_list = cut_ratio_list
        self.user_id_list = user_id_list
        self.activity_id_list = activity_id_list
        if self.task == 'heart_rate_estimation':
            self.label_loader = self.heart_rate_loader
        elif self.task == 'pose_estimation':
            self.label_loader = self.optitrack_loader
        else:
            self.label_loader = None # by default, the label loader is None, and we will consider the task as activity recognition
        
        if self.modality == 'UWB':
            self.data_loader = self.uwb_loader
        elif self.modality == 'mmWave':
            self.data_loader = self.mmWave_loader
        elif self.modality == 'WiFi1' or self.modality == 'WiFi2' or self.modality == 'WiFi3' or self.modality == 'WiFi4':
            self.data_loader = self.csi_loader
        else:
            raise ValueError("Invalid modality")
        
    def data_loading(self, indexes):
        start_index, end_index = indexes
        data_list = []
        label_list = []
        user_id_list = []
        activity_id_list = []
        for i in tqdm(range(start_index, end_index)):
            data_path = self.data_path_list[i]
            cut_ratios = self.cut_ratio_list[i]
            user_id = self.user_id_list[i]
            activity_id = self.activity_id_list[i]
            data = self.data_loader(data_path, cut_ratios)
            if self.label_loader is not None:
                label_path = self.label_path_list[i]
                label = self.label_loader(label_path, cut_ratios)
            else:
                label = self.activity_id_list[i]
                # we need to repeat the label (activity) for the number of segments
                label = [label for _ in range(len(data))]
            data_list.append(data)
            label_list.append(label)
            # repeat the user id and activity id for the number of segments
            user_id_list.append([user_id for _ in range(len(data))])
            activity_id_list.append([activity_id for _ in range(len(data))])
        return data_list, label_list, user_id_list, activity_id_list
            
    
    def csi_loader(self, file_path, cut_ratios, csi_shape=(2,114), mac_filter='A4:A9:30:B1:AF:7D'):
        # (2, 114): two receive antennas and 114 subcarriers from the transmitter with mac address 'A4:A9:30:B1:AF:7D'
        # Only one antenna is activated for the transmitter
        csi_all, _, header_all = read_csi_data(file_path, csi_shape, mac_filter)
        timestamps = [header['timestamp'] for header in header_all]
        recording_duration_us = timestamps[-1] - timestamps[0]
        recording_duration_s = recording_duration_us / 1e6
        # sampling_rate = len(csi_all) / recording_duration_s
        sampling_rate = 100
        csi_matrix_all = np.array(csi_all)
        csi_matrix_all = np.transpose(csi_matrix_all, (0, 2, 1))
        csi_matrix_all = np.expand_dims(csi_matrix_all, axis=-1) # shape: num_frames, num_subcarriers (114), num_rx (2), num_tx (1)
        
        if self.preprocess:
            csi_matrix_all, motion_statistics = preprocess_CSI(csi_matrix_all, sampling_rate)

        CSI_matrix_list = []
        # cut the csi data into different segments based on the cut ratios
        total_num_frames, num_subcarriers, num_rx, num_tx = csi_matrix_all.shape
        for i in range(1, len(cut_ratios)):
            start_ratio = cut_ratios[i-1]
            end_ratio = cut_ratios[i]
            start_frame = int(start_ratio * total_num_frames)
            end_frame = int(end_ratio * total_num_frames)
            csi_matrix = csi_matrix_all[start_frame:end_frame]
            num_frames = csi_matrix.shape[0]
            if num_frames < self.time_length:
                num_frames = self.time_length
            csi = torch.tensor(csi_matrix)  # shape: (frames, subcarriers, Rx, Tx)
            csi = csi.view(csi.shape[0], csi.shape[1], csi.shape[2]*csi.shape[3])  # shape: (frames, subcarriers, Rx*Tx)
            if self.format == 'polar':
                csi_amp = torch.abs(csi)
                csi = torch.cat((csi_amp, torch.angle(csi)), dim=-1) # shape: (frames, subcarriers, Rx*Tx*2)
                padded_csi = torch.zeros(num_frames, num_subcarriers, num_rx*num_tx*2) 
                padded_csi[:csi.shape[0], :csi.shape[1], :csi.shape[2]*2] = csi
                if self.time_length is not None:
                    if num_frames > self.time_length:
                        # select the middle part of the csi data
                        start_index = (num_frames - self.time_length) // 2
                        padded_csi = padded_csi[start_index:start_index+self.time_length, :, :]
                    else:
                        padded_csi = padded_csi[-self.time_length:, :, :]
            elif self.format == 'cartesian':
                csi = torch.cat((csi.real, csi.imag), dim=-1)
                padded_csi = torch.zeros(num_frames, num_subcarriers, num_rx*num_tx*2)
                padded_csi[:csi.shape[0], :csi.shape[1], :csi.shape[2]*2] = csi
                if self.time_length is not None:
                    if num_frames > self.time_length:
                        # select the middle part of the csi data
                        start_index = (num_frames - self.time_length) // 2
                        padded_csi = padded_csi[start_index:start_index+self.time_length, :, :]
                    else:
                        padded_csi = padded_csi[:self.time_length, :, :]
            else:
                padded_csi = torch.zeros((num_frames, num_subcarriers, num_rx*num_tx), dtype=torch.complex64)
                padded_csi[:csi.shape[0], :csi.shape[1], :csi.shape[2]] = csi
                if self.time_length is not None:
                    if num_frames > self.time_length:
                        # select the middle part of the csi data
                        start_index = (num_frames - self.time_length) // 2
                        padded_csi = padded_csi[start_index:start_index+self.time_length, :, :]
                    else:
                        padded_csi = padded_csi[:self.time_length, :, :]
            if self.format == 'dfs':
                freq_bin, ticks, freq_time_prof = get_csi_dfs(padded_csi, samp_rate = 100, window_size = 25, window_step = 4) 
                # shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
                CSI_matrix_list.append(freq_time_prof)
            else:
                CSI_matrix_list.append(padded_csi)
        return CSI_matrix_list 

    def uwb_loader(self, file_path, cut_ratios):
        data_dicts = []
        with open(file_path, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    data_dicts.append(data)
                except EOFError:
                    break
        UWB_data = []
        # timestamps = []
        for index in range(len(data_dicts)):
            frame = data_dicts[index]['frame']
            timestamp = data_dicts[index]['timestamp']
            frame = np.array(frame)
            frame = frame[:800]  # the max range is 4 meters; start from the '25' is to remove the Tx direct path (Leakage)
            fc = 7.29e9 # Lower pulse generator setting
            fs = 23.328e9 # X4 sampling rate
            csine = np.exp(-1j*fc/fs*2*np.pi*np.arange(len(frame)))
            cframe = frame*csine
            UWB_data.append(cframe)
            # timestamps.append(timestamp)
            
        # cut the UWB data into different segments based on the cut ratios
        uwb_data_list = []
        total_num_frames = len(UWB_data)
        for i in range(1, len(cut_ratios)):
            start_ratio = cut_ratios[i-1]
            end_ratio = cut_ratios[i]
            start_frame = int(start_ratio * total_num_frames)
            end_frame = int(end_ratio * total_num_frames)
            data_temp = np.array(UWB_data[start_frame:end_frame], dtype=np.complex64)
            time_length_temp = data_temp.shape[0]
            if time_length_temp < self.time_length:
                data_temp = np.concatenate((data_temp, np.zeros((self.time_length-time_length_temp, data_temp.shape[1]), dtype=np.complex64)), axis=0)
            elif time_length_temp > self.time_length:
                data_temp = data_temp[-self.time_length:]
            data_temp = torch.tensor(data_temp, dtype=torch.complex64)
            if self.format == 'polar':
                amp = torch.abs(data_temp)
                data_temp = torch.stack((amp, torch.angle(data_temp)), dim=-1)  # shape: T,800, 2
            elif self.format == 'cartesian':
                data_temp = torch.stack((data_temp.real, data_temp.imag), dim=-1)  # shape: T, 800, 2
            elif self.format == 'complex':
                pass
            else:
                raise ValueError('Invalid format')
            uwb_data_list.append(data_temp)
        return uwb_data_list

    def mmWave_loader(self, file_path, cut_ratios): 
        mat_data = loadmat(file_path, squeeze_me=True)
        recs = mat_data['recs']  # Rx*Tx (20*20), fast time (100 ADC), slow time (frames) :: complex128
        recs = np.transpose(recs, (2, 0, 1))  # shape: slow time (frames), Rx*Tx, fast time (100 ADC)
        
        mmWave_data_list = []
        total_num_frames = recs.shape[0]
        for i in range(1, len(cut_ratios)):
            start_ratio = cut_ratios[i-1]
            end_ratio = cut_ratios[i]
            start_frame = int(start_ratio * total_num_frames)
            end_frame = int(end_ratio * total_num_frames)
            data_temp = torch.tensor(recs[start_frame:end_frame], dtype=torch.complex64)
            time_length_temp = data_temp.shape[0]
            if time_length_temp < self.time_length:
                data_temp = torch.cat((data_temp, torch.zeros((self.time_length-time_length_temp, data_temp.shape[1], data_temp.shape[2]), dtype=torch.complex64)), dim=0)
            elif time_length_temp > self.time_length:
                data_temp = data_temp[-self.time_length:]
            if self.format == 'polar': # shape: T, 20*20, 100, 2
                amp = torch.abs(data_temp)
                data_temp = torch.stack((amp, torch.angle(data_temp)), dim=-1)
            elif self.format == 'cartesian': # shape: T, 20*20, 100, 2
                data_temp = torch.stack((data_temp.real, data_temp.imag), dim=-1)
            elif self.format == 'complex': # shape: T, 20*20, 100
                pass
            else:
                raise ValueError('Invalid format')
            mmWave_data_list.append(data_temp)
        return mmWave_data_list

    def heart_rate_loader(self, file_path, cut_ratios):
        data_dicts = []
        with open(file_path, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    data_dicts.append(data)
                except EOFError:
                    break
    
        heart_rate_all = []
        # timestamps = []
        for data in data_dicts:
            heart_rate_all.append(data['data'])
            # timestamps.append(data['timestamp'])
        heart_rate_all = np.array(heart_rate_all)
        heart_rate_list = []
        total_num_frames = len(heart_rate_all)
        for i in range(1, len(cut_ratios)):
            start_ratio = cut_ratios[i-1]
            end_ratio = cut_ratios[i]
            start_frame = int(start_ratio * total_num_frames)
            end_frame = int(end_ratio * total_num_frames)
            data_temp = torch.tensor(heart_rate_all[start_frame:end_frame], dtype=torch.float32)
            time_length_temp = data_temp.shape[0]
            if time_length_temp < self.num_classes:
                data_temp = torch.cat((data_temp, torch.zeros(self.num_classes-time_length_temp, dtype=torch.float32)), dim=0)
            elif time_length_temp > self.num_classes:
                data_temp = data_temp[-self.num_classes:]
            heart_rate_list.append(data_temp)
        return heart_rate_list
    
    def optitrack_loader(self, file_path, cut_ratios):
        optitrack_skeleton = OptitrackSkeleton(file_path)
        # we will return the last pose in each segment as the label
        bone_positions_all, _ = optitrack_skeleton.get_skeleton_data_all()  # shape: num_frames, num_bones, 3
        total_num_frames = bone_positions_all.shape[0]
        skeleton_list = []
        for i in range(1, len(cut_ratios)):
            start_ratio = cut_ratios[i-1]
            end_ratio = cut_ratios[i]
            start_frame = int(start_ratio * total_num_frames)
            end_frame = int(end_ratio * total_num_frames)
            data_temp = bone_positions_all[start_frame:end_frame]
            last_frame = data_temp[-1] # shape: num_bones, 3
            data_temp = torch.tensor(last_frame, dtype=torch.float32)
            skeleton_list.append(data_temp)
        return skeleton_list
        
    def get_len(self):
        return len(self.data_path_list)


class OctonetMini(Dataset):
    def __init__(self, config):
        self.config = config
        self.modality = config['modality']   # the input modality: UWB, mmWave, WiFi1, WiFi2, WiFi3, WiFi4
        self.task = config['task']           # the target task: activity_recognition, heart_rate_estimation, pose_estimation
        self.dataset_path = config['dataset_path']
        self.num_workers = config['num_workers']
        self.user_selected = config["user_list"]  # the selected user list:: [1,2,3,4,5,6,7,8,9,10]
        self.activity_slected = config["activity_list"]  # the selected activity: tuple (start activity_id, end activity_id) :: 62 activities in total
        # loading the meta info dictionary
        with open(self.dataset_path + '/' + 'meta_data_with_cut_points.pkl', 'rb') as f:
            self.meta_data_dict = pickle.load(f)
            
        self.activity = list(self.meta_data_dict['activity'].values())                           # the activity label
        self.user_id = list(self.meta_data_dict['user_id'].values())                             # the user id
        self.recording_time = list(self.meta_data_dict['recording_time'].values())               # the recording time
        self.heart_rate_data_path = list(self.meta_data_dict['heart_rate_data_path'].values())   # the heart rate data path for the heart rate estimation task
        self.optitrack_data_path = list(self.meta_data_dict['optitrack_data_path'].values())     # the optitrack data path for the pose estimation task
        self.cut_points = self.meta_data_dict['cut_points']                                      # the cut points to segment the optitrack data into different segments within one recording
        self.cut_ratio = self.meta_data_dict['cut_ratio']                                        # the cut ratio is from the cut points to segment the other modality into different segments within one recording
        self.threshold = self.meta_data_dict['threshold']                                        # the speed threshold that determines the low-speed region for generating the cut points
        
        self.uwb_data_path = list(self.meta_data_dict['uwb_data_path'].values())                
        self.vayyar_data_path = list(self.meta_data_dict['vayyar_data_path'].values())
        self.wifi1_data_path = list(self.meta_data_dict['wifi1_data_path'].values())
        self.wifi2_data_path = list(self.meta_data_dict['wifi2_data_path'].values())
        self.wifi3_data_path = list(self.meta_data_dict['wifi3_data_path'].values())
        self.wifi4_data_path = list(self.meta_data_dict['wifi4_data_path'].values())
        
        if self.modality == 'UWB':
            self.data_path = self.uwb_data_path
        elif self.modality == 'mmWave':
            self.data_path = self.vayyar_data_path
        elif self.modality == 'WiFi1':
            self.data_path = self.wifi1_data_path
        elif self.modality == 'WiFi2':
            self.data_path = self.wifi2_data_path
        elif self.modality == 'WiFi3':
            self.data_path = self.wifi3_data_path
        elif self.modality == 'WiFi4':
            self.data_path = self.wifi4_data_path
        else:
            raise ValueError("Invalid modality")
        
        if self.task == 'heart_rate_estimation':
            print("The task is heart rate estimation")
            self.label_path = self.heart_rate_data_path
        elif self.task == 'pose_estimation':
            print("The task is pose estimation")
            self.label_path = self.optitrack_data_path
            # we will shift the cut_ratio a bit to make sure the estimated pose is the pose during the activity to make it diverse
            for i in range(len(self.cut_ratio)):
                if len(self.cut_ratio[i]) != 0:
                    shift = 0.5 * self.cut_ratio[i][0]
                    self.cut_ratio[i] = [max(0, ratio-shift) for ratio in self.cut_ratio[i]]
        else:
            print("By default, the task is activity recognition")
            self.label_path = None     # bydefault, the label path is None; and we will consider the task as activity recognition
        
        # generate the mapping from the activity label to the one-hot encoding
        act_valid_label = [act for act in self.activity if 'testing' not in act and 'none' not in act]
        self.activity_types = list(set(act_valid_label))
        self.activity2actID = {act: i for i, act in enumerate(self.activity_types)}
        self.actID2activity = {i: act for act, i in self.activity2actID.items()}
        
        # we will filter out the samples that are not in the valid activity, not have the label path, not have the data path, and not have the cut ratio
        self.activity_based_valid_index = [i for i, act in enumerate(self.activity) if 'testing' not in act and 'none' not in act]
        self.data_path_based_valid_index = [i for i, path in enumerate(self.data_path) if str(path) != 'nan']
        if self.label_path is not None:   # which means the task is pose estimation or heart rate estimation
            self.label_path_based_valid_index = [i for i, path in enumerate(self.label_path) if str(path) != 'nan']
        else:   # which means the task is activity recognition
            self.label_path_based_valid_index = self.activity_based_valid_index
        self.cut_ratio_based_valid_index = [i for i, ratio in enumerate(self.cut_ratio) if len(ratio) != 0]
        self.valid_index = list(set(self.activity_based_valid_index) & set(self.data_path_based_valid_index) & set(self.label_path_based_valid_index) & set(self.cut_ratio_based_valid_index))
        
        # valid the samples:
        self.activity = [self.activity[i] for i in self.valid_index]
        self.activity_id = [self.activity2actID[act] for act in self.activity]
        self.user_id = [self.user_id[i] for i in self.valid_index]
        self.recording_time = [self.recording_time[i] for i in self.valid_index]
        self.data_path = [self.data_path[i] for i in self.valid_index]
        if self.label_path is not None:
            self.label_path = [self.label_path[i] for i in self.valid_index]
        self.cut_points = [self.cut_points[i] for i in self.valid_index]
        self.cut_ratio = [self.cut_ratio[i] for i in self.valid_index]
        
        # select the samples based on the user list and the activity list
        self.selected_index = [i for i, act_id in enumerate(self.activity_id) if act_id >= self.activity_slected[0] and act_id <= self.activity_slected[1] and self.user_id[i] in self.user_selected]
        self.activity = [self.activity[i] for i in self.selected_index]
        self.activity_id = [self.activity_id[i] for i in self.selected_index]
        self.user_id = [self.user_id[i] for i in self.selected_index]
        self.recording_time = [self.recording_time[i] for i in self.selected_index]
        self.data_path = [self.dataset_path + '/' + self.data_path[i] for i in self.selected_index]
        if self.modality == 'WiFi1' or self.modality == 'WiFi2' or self.modality == 'WiFi3' or self.modality == 'WiFi4':
            self.data_path = [path + '/' + name for path in self.data_path for name in os.listdir(path) if name.endswith('.csi')]       
        if self.label_path is not None:
            self.label_path = [self.dataset_path + '/' + self.label_path[i] for i in self.selected_index]
        else:
            self.label_path = None
        self.cut_points = [self.cut_points[i] for i in self.selected_index]
        self.cut_ratio = [self.cut_ratio[i] for i in self.selected_index]
        
        self.data_loader = multi_processing_data_loader(config, self.data_path, self.label_path, self.cut_ratio, self.user_id, self.activity_id)
        num_files = self.data_loader.get_len()
        self.data = []
        self.label = []
        self.user_id = []
        self.activity_id = []
            
        print("num_files: ", num_files)
        # split the data into num_workers parts
        index_list = []   # each element is a tuple (start_index, end_index)
        for i in range(self.num_workers):
            start_index = i * num_files // self.num_workers
            end_index = (i + 1) * num_files // self.num_workers
            index_list.append((start_index, end_index))
        print("index_list: ", index_list)
            
        # multi-processing data loading
        with Pool(self.num_workers) as p:
            results = p.map(self.data_loader.data_loading, index_list)  # each element is a tuple (data_list, label_list, user_id_list, activity_id_list)
        for result in results:
            data_list, label_list, user_id_list, activity_id_list = result
            for i in range(len(data_list)):
                self.data.extend(data_list[i])
                self.label.extend(label_list[i])
                self.user_id.extend(user_id_list[i])
                self.activity_id.extend(activity_id_list[i])
        print("The data loading is finished")
        print("The number of samples: ", len(self.data))

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.label[index]
        user_id = self.user_id[index]
        activity_id = self.activity_id[index]
        return sample, label, user_id,activity_id
            
    def __len__(self):
        return len(self.data)
        

# converting the data shape to the shape that the model needs
class OctooNetMini_data_shape_converter:
    def __init__(self,config):
        self.model_input_shape = config['model_input_shape'] # BCHW, BLC, BTCHW, B2CNFT 
        # B: batch size, C: channel, H: height, W: width, T: time length, N: number of (WiFi CSI) links, F: frequency bins
        self.format = config['format']
        self.modality = config['modality'] # the input modality: UWB, mmWave, WiFi1, WiFi2, WiFi3, WiFi4
    
    def shape_convert(self,batch):
        if self.modality == 'UWB': # shape: [batch_size, frames, ADC_sample (800), *]
            if self.format == 'complex': # shape: [batch_size, frames, ADC_sample (800)]
                if self.model_input_shape == 'BCHW':
                    batch = batch.permute(0,2,1)  # shape: [batch_size, ADC_sample (800), frames]
                    batch = batch.unsqueeze(1)  # shape: [batch_size, 1, ADC_sample (800), frames]
                elif self.model_input_shape == 'BLC':
                    batch = batch.permute(0,2,1)  # shape: [batch_size, ADC_sample (800), frames]
                elif self.model_input_shape == 'BTCHW' or self.model_input_shape == 'B2CNFT':
                    raise ValueError("The config.model_input_shape 'BTCHW' or 'B2CNFT'  not supported by config.modality 'UWB'.")
            elif self.format == 'polar' or self.format == 'cartesian': # shape: [batch_size, frames, ADC_sample (800), 2]
                if self.model_input_shape == 'BCHW':
                    batch = batch.permute(0,3,2,1)  # shape: [batch_size, 2, ADC_sample (800), frames]
                elif self.model_input_shape == 'BLC':
                    batch = batch.permute(0,2,1,3) # shape: [batch_size, ADC_sample (800), 2, frames]
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]) # shape: [batch_size, ADC_sample (800), 2*frames] 
            else:
                raise ValueError("The config.format must be one of 'polar', 'cartesian', 'complex'.")
        elif self.modality == 'mmWave': # shape: [batch_size, frames, Rx*Tx, ADC_sample (100), *]
            if self.format == 'complex': # shape: [batch_size, frames, Rx*Tx, ADC_sample (100)]
                if self.model_input_shape == 'BCHW':
                    batch = batch.permute(0,2,3,1) # shape: [batch_size, Rx*Tx, ADC_sample (100), frames]
                elif self.model_input_shape == 'BLC':
                    batch = batch.permute(0,1,3,2) # shape: [batch_size, frames, ADC_sample (100), Rx*Tx]
                    # merge the frames and ADC_sample into one dimension
                    batch = batch.view(batch.shape[0], batch.shape[1]*batch.shape[2], batch.shape[3]) # shape: [batch_size, frames*ADC_sample (100), Rx*Tx]
                elif self.model_input_shape == 'BTCHW' or self.model_input_shape == 'B2CNFT':
                    raise ValueError("The config.model_input_shape 'BTCHW' or 'B2CNFT'  not supported by config.modality 'mmWave'.")
            elif self.format == 'polar' or self.format == 'cartesian': # shape: [batch_size, frames, Rx*Tx, ADC_sample (100), 2]
                if self.model_input_shape == 'BCHW':
                    batch = batch.permute(0,2,4,3,1) # shape: [batch_size, Rx*Tx, 2, ADC_sample (100), frames]
                    # merge the Rx*Tx and 2 into one dimension
                    batch = batch.view(batch.shape[0], batch.shape[1]*batch.shape[2], batch.shape[3], batch.shape[4]) # shape: [batch_size, Rx*Tx*2, ADC_sample (100), frames]
                elif self.model_input_shape == 'BLC':
                    batch = batch.permute(0,1,3,2,4) # shape: [batch_size, frames, ADC_sample (100), Rx*Tx, 2]
                    batch = batch.view(batch.shape[0], batch.shape[1]*batch.shape[2], batch.shape[3]*batch.shape[4]) # shape: [batch_size, frames*ADC_sample (100), Rx*Tx*2]
            else:
                raise ValueError("The config.format must be one of 'polar', 'cartesian', 'complex'.")
        elif self.modality == 'WiFi1' or self.modality == 'WiFi2' or self.modality == 'WiFi3' or self.modality == 'WiFi4': # shape: [batch_size, frames, subcarriers, Rx*Tx*]
            if self.format == 'dfs':    # shape of the input batch: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
                if self.model_input_shape == 'BCHW':
                    # merge the last three dimensions of the batch into one dimension: num_subcarriers * Rx*Tx * 2
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]*batch.shape[4]*2) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers * Rx*Tx * 2]
                    # transpose the last dim to the second dim as channel
                    batch = batch.permute(0,3,1,2)
                elif self.model_input_shape == 'BCHW-C': # the complex version of the BCHW
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
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]*batch.shape[4]*2) # shape: [batch_size, Time_bins, freq_bins*num_subcarriers * Rx*Tx * 2]
                elif self.model_input_shape == 'BTCHW': # for baseline: Widar3.0 model
                    # merge the last two dimensions of the batch into one dimension: Rx*Tx * 2
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]*2) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx * 2]
                    # transpose the last dim to the third dim as channel
                    batch = batch.permute(0,1,4,2,3)  # shape: [batch_size, Time_bins, Rx*Tx * 2, freq_bins, num_subcarriers]
                elif self.model_input_shape == 'B2CNFT':  # for baseline: SLNet
                    batch = batch.permute(0,5,3,4,2,1)  # shape: [batch_size, 2, num_subcarriers, Rx*Tx, freq_bins, Time_bins]
            elif self.format == 'polar' or self.format == 'cartesian': 
                # shape of the input batch: [batch_size, frames, subcarriers, Rx*Tx*2]
                if self.model_input_shape == 'BCHW':
                    batch = batch.permute(0,3,1,2) # shape: [batch_size, Rx*Tx*2, frames, subcarriers]
                elif self.model_input_shape == 'BLC':
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]) # shape: [batch_size, frames, subcarriers * Rx*Tx * 2]
                elif self.model_input_shape == 'BTCHW' or self.model_input_shape == 'B2CNFT':
                    raise ValueError("The config.model_input_shape 'BTCHW' or 'B2CNFT'  only supported by config.format 'dfs'.")
                else:
                    raise ValueError("The config.model_input_shape must be one of 'BCHW', 'BLC'.")
            elif self.format == 'complex':
                # shape of the input batch: [batch_size, frames, subcarriers, Rx*Tx]  in tprch.complex64
                if self.model_input_shape == 'BCHW':
                    batch = batch.permute(0,3,1,2) # shape: [batch_size, Rx*Tx, frames, subcarriers]
                elif self.model_input_shape == 'BLC':
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]) # shape: [batch_size, frames, subcarriers * Rx*Tx]
                elif self.model_input_shape == 'BTCHW' or self.model_input_shape == 'B2CNFT':
                    raise ValueError("The config.model_input_shape 'BTCHW' or 'B2CNFT'  only supported by config.format 'dfs'.")
                else:
                    raise ValueError("The config.model_input_shape must be one of 'BCHW', 'BLC'.")
            else:
                raise ValueError("The config.format must be one of 'polar', 'cartesian', 'dfs', 'complex'.")
        else:
            raise ValueError("Invalid modality")
        return batch

def make_OctooNetMini_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd, num_workers = 0):
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

