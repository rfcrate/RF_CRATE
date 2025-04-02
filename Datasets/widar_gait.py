import glob, os, math
from random import shuffle, seed
from CSIKit.reader import IWLBeamformReader
from torch.utils.data import Dataset, DataLoader
from .utils import get_CSI, preprocess_CSI, get_dfs, get_csi_dfs
import pandas as pd
from tqdm import tqdm
import torch
import yaml
import argparse
import pickle
from multiprocessing import Pool
from multiprocessing import get_context
import numpy as np


RAW_PREFIX = 'CSI'  # this is the folder where the raw data is stored

def WidarGaitDatabase(config):
    dataset_path = config["dataset_path"]
    user_list = config["user_list"]  # the selected user list
    track_list = config["track_list"]       # the selected track
    select_rx = config["select_rx"]    # the selected rx lis

    df = pd.read_csv(os.path.join(dataset_path, "metas.csv"))  # file_name  user_id  track_id  rep_id  rx_id
    file_names = df['file_name'].values
    user_ids = df['user_id'].values
    track_ids = df['track_id'].values
    receiver_ids = df['rx_id'].values
    
    # keep the rows that match the selected user, track, and rx
    selected_rows = []
    for i in range(len(file_names)):
        if user_ids[i] in user_list and track_ids[i] in track_list and receiver_ids[i] in select_rx:
            selected_rows.append(i)
    
    file_names = file_names[selected_rows]
    # add the prefix to the file names
    file_names = [os.path.join(RAW_PREFIX, file_name) for file_name in file_names]
    user_ids = user_ids[selected_rows]
    track_ids = track_ids[selected_rows]
    receiver_ids = receiver_ids[selected_rows]
    return file_names, user_ids, track_ids, receiver_ids


class multi_processing_data_loader:
    def __init__(self, file_name_list, config, label_list=None, user_list=None, track_list=None, select_rx=None):
        self.file_name_list = file_name_list
        self.preprocess = config['preprocess']
        self.format = config['format']
        self.time_length = config['time_length']
        self.tqdm_disable = config['tqdm_disable']
        self.label_list = label_list
        self.user_list = user_list
        self.track_list = track_list
        self.select_rx = select_rx
        self.data_path = config['dataset_path']

    def load_csi_data(self, indexes):
        csi_data_list = []
        if self.format == 'dense_dfs' or self.format == 'dense_dfs_amp': # reture the stored dfs (with stft.window_step = 10) data with shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
            start_index, end_index = indexes
            loading_files = self.file_name_list[start_index:end_index]
            for file_name in tqdm(loading_files, disable=self.tqdm_disable):
                # replace the 'CSI' with 'DFS' in the file name and 'dat' with 'npy'
                dfs_file_name = file_name.replace('CSI', 'DFS')
                dfs_file_name = dfs_file_name.replace('dat', 'npy')
                if os.path.exists(dfs_file_name):
                    csi_data = np.load(dfs_file_name)
                    time_bins = csi_data.shape[0]
                    if time_bins < 80: # skip the csi data with short time bins
                        csi_data_list.append(None)
                        continue
                    if time_bins <= self.time_length // 10: # considering the time length of the csi data under the window step of 10
                        # padding the csi data to the same length
                        padded_csi = np.zeros((self.time_length // 10, csi_data.shape[1], csi_data.shape[2], csi_data.shape[3], csi_data.shape[4]))
                        padded_csi[:time_bins, :, :, :, :] = csi_data
                    else:  # select the middle part of the csi data
                        start_index = (time_bins - self.time_length // 10) // 2
                        padded_csi = csi_data[start_index:start_index+self.time_length // 10, :, :, :, :]
                    csi_data_list.append(torch.tensor(padded_csi))
                else:
                    csi_data_list.append(None)
        else:
            my_reader = IWLBeamformReader()
            start_index, end_index = indexes
            loading_files = self.file_name_list[start_index:end_index]
            for file_name in tqdm(loading_files, disable=self.tqdm_disable):
                csi_data = my_reader.read_file(file_name)  
                csi_matrix, _, _ = get_CSI(csi_data)  # shape: (frames, subcarriers, Rx, Tx)
                if csi_matrix is None:
                    csi_data_list.append(None)
                    continue
                if self.preprocess:   # phase cleaning and filering
                    csi_matrix, _ = preprocess_CSI(csi_matrix)
                
                num_frames, num_subcarriers, num_rx, num_tx = csi_matrix.shape
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
                            padded_csi = padded_csi[:self.time_length, :, :]
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
                    freq_bin, ticks, freq_time_prof = get_csi_dfs(padded_csi, samp_rate = 1000, window_size = 256, window_step = 35) 
                    # shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
                    csi_data_list.append(freq_time_prof)
                else:
                    csi_data_list.append(padded_csi)
        return (csi_data_list, indexes)
    
    def get_valid_csi_data_files(self, indexes):
        # retrun the file names of the valid csi data: the csi data with None is not valid and the csi data with too short frames is not valid
        start_index, end_index = indexes
        loading_files = self.file_name_list[start_index:end_index]
        my_reader = IWLBeamformReader()
        valid_files = []
        valid_labels = []
        valid_track_ids = []
        valid_receiver_ids = []
        valid_user_ids = []
        if self.label_list is None:
            assert False, "The label list is None."
        else:
            label_list = self.label_list[start_index:end_index]
        for index, file_name in enumerate(tqdm(loading_files, disable=self.tqdm_disable)):
            if self.format == 'dense_dfs' or self.format == 'dense_dfs_amp': # reture the stored dfs (with stft.window_step = 10) data with shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
                dfs_file_name = file_name.replace('CSI', 'DFS')
                dfs_file_name = dfs_file_name.replace('dat', 'npy')
                # print(dfs_file_name)
                if os.path.exists(dfs_file_name):
                    csi_data = np.load(dfs_file_name)
                    time_bins = csi_data.shape[0]
                    if time_bins < 80: # skip the csi data with short time bins
                        continue
                    valid_files.append(file_name)
                    valid_labels.append(label_list[index])
                    valid_track_ids.append(self.track_list[start_index+index])
                    valid_receiver_ids.append(self.select_rx[start_index+index])
                    valid_user_ids.append(self.user_list[start_index+index])
                continue
            else:
                if self.preprocess and os.path.exists(self.data_path + '/CSI_Processed'):
                    # if the 'CSI_Processed' folder is used, we need to change the valid_sample_list_file_name
                    # check the folder 'CSI_Processed' exists or not
                    file_name = file_name.replace('CSI', 'CSI_Processed')  # if we have preprocessed and stored the csi data in CSI_Processed folder, we use the preprocessed data
                    file_name = file_name.replace('dat', 'npy')
                    if not os.path.exists(file_name):
                        continue
                    csi_matrix = np.load(file_name)
                    if csi_matrix is None:
                        continue
                    num_frames, _, _, _ = csi_matrix.shape
                    if num_frames < 1000:
                        continue
                    valid_files.append(file_name)
                    valid_labels.append(label_list[index])
                    valid_track_ids.append(self.track_list[start_index+index])
                    valid_receiver_ids.append(self.select_rx[start_index+index])
                    valid_user_ids.append(self.user_list[start_index+index])
                else:
                    csi_data = my_reader.read_file(file_name)  
                    csi_matrix, _, _ = get_CSI(csi_data)
                    if csi_matrix is None:
                        continue
                    # skip the csi with short frames
                    num_frames, _, _, _ = csi_matrix.shape
                    # if num_frames < 1000:
                    #     continue
                    valid_files.append(file_name)
                    valid_labels.append(label_list[index])
                    valid_track_ids.append(self.track_list[start_index+index])
                    valid_receiver_ids.append(self.select_rx[start_index+index])
                    valid_user_ids.append(self.user_list[start_index+index])
        return valid_files, valid_labels, valid_track_ids, valid_receiver_ids, valid_user_ids
    
    def get_len(self):
        return len(self.file_name_list)

class WidarGait_Dataset(Dataset):
    def __init__(self,config):
        self.config = config
        self.file_names, self.user_ids, self.track_ids, self.receiver_ids = WidarGaitDatabase(config)
        self.file_paths = []
        self.label_mapping = {}
        self.labels = []
        self.parse_paths_labels()
        self.my_reader = IWLBeamformReader()
        self.csi_device_info = None
        self.num_workers = config['num_workers']
        self.preload = config['preload']
        self.tqdm_disable = config['tqdm_disable']
        self.preprocess = config['preprocess']
        self.batch_size = config['batch_size']
        self.format = config['format']
        self.time_length = config['time_length']
        self.dataset_path = config['dataset_path']
        
        # get the csi device info
        print("Getting the CSI device info...")
        csi_data = self.my_reader.read_file(self.file_paths[0])
        if self.csi_device_info is None:
            self.csi_device_info = csi_data.get_metadata()

        if self.preload: # load all the csi data into memory
            self.data_loader = multi_processing_data_loader(self.file_paths, config)
            print("Preloading the CSI data...")
            self.data = [0 for i in range(len(self.file_paths))]
            num_files = self.data_loader.get_len()
            # split the data into num_workers parts
            index_list = []   # each element is a tuple (start_index, end_index)
            for i in range(self.num_workers):
                start_index = i * num_files // self.num_workers
                end_index = (i + 1) * num_files // self.num_workers
                index_list.append((start_index, end_index))
            # with Pool(self.num_workers) as p:
            with get_context("spawn").Pool(self.num_workers) as p:
                for csi_data, index_range in p.imap_unordered(self.data_loader.load_csi_data, index_list, chunksize=1):
                    start_index, end_index = index_range
                    self.data[start_index:end_index] = csi_data
                # release pool resources
                p.close()
                p.join()
            print("Preloading done.")
            # fillter out the csi data with None
            none_values_index = [i for i, x in enumerate(self.data) if x is None]
            self.file_paths = [i for j, i in enumerate(self.file_paths) if j not in none_values_index]
            self.labels = [i for j, i in enumerate(self.labels) if j not in none_values_index]
            self.data = [i for j, i in enumerate(self.data) if j not in none_values_index]
            self.track_ids = [i for j, i in enumerate(self.track_ids) if j not in none_values_index]
            self.receiver_ids = [i for j, i in enumerate(self.receiver_ids) if j not in none_values_index]
            self.user_ids = [i for j, i in enumerate(self.user_ids) if j not in none_values_index]
            print("The number of valid CSI data: ", len(self.data))
        else: 
            print("Using the lazy loading mode.")
            print("The number of instances before data sanitization: ", len(self.file_paths))
            print("Sanitizing the CSI data...")
            
            # the valid sample list file name is based on the config file   
            user_list_str = ''.join(str(e) for e in sorted(config['user_list']))
            track_list_str = ''.join(str(e) for e in sorted(config['track_list']))
            select_rx_str = ''.join(str(e) for e in sorted(config['select_rx']))
            
            valid_sample_list_file_name = config['dataset_name'] +  "_user" + user_list_str + "_track_" + track_list_str + "_rx_" + select_rx_str + '_valid_file_label.pkl'
            valid_sample_list_file_name = 'Datasets/valid_file_lists/' + valid_sample_list_file_name
            if self.format == 'dense_dfs' or self.format == 'dense_dfs_amp': # the dense (stft window size: 256, step size: 10.) dfs data should be prepared and stored in the DFS folder
                if os.path.exists(self.dataset_path + '/DFS'):
                    valid_sample_list_file_name = valid_sample_list_file_name.replace('.pkl', '_dense_dfs.pkl')
                else:
                    raise ValueError("The dense dfs data is not found. Please prepare the dense dfs data first.")
            else:
                # if the 'CSI_Processed' folder is used, we can directly load the processed CSI data (.npy format) to save time.
                if os.path.exists(self.dataset_path + '/CSI_Processed'):
                    valid_sample_list_file_name = valid_sample_list_file_name.replace('.pkl', '_processed_CSI.pkl')
                else: # we need to load the raw CSI data in the get_item function and preprocess the data
                    pass
                
            if os.path.exists(valid_sample_list_file_name):
                print("The valid sample list file exists.")
                with open(valid_sample_list_file_name, 'rb') as f:
                    valid_sample_list = pickle.load(f)
                    self.file_paths = valid_sample_list['file_paths']
                    self.labels = valid_sample_list['labels']
                    self.track_ids = valid_sample_list['track_ids']
                    self.receiver_ids = valid_sample_list['receiver_ids']
                    self.user_ids = valid_sample_list['user_ids']
                    print("The valid file paths and labels are loaded from: ", valid_sample_list_file_name)
                    print("The number of valid CSI data after sanitizing: ", len(self.file_paths))
            else:
                self.data_loader = multi_processing_data_loader(self.file_paths, 
                                                                config, 
                                                                label_list=self.labels, 
                                                                user_list=self.user_ids, 
                                                                track_list=self.track_ids, 
                                                                select_rx=self.receiver_ids)
                num_files = self.data_loader.get_len()
                # split the data into num_workers parts
                index_list = []
                valid_files_list = []
                valid_labels_list = []
                valid_track_ids_list = []
                valid_receiver_ids_list = []
                valid_user_ids_list = []
                # using multi-processing to get the valid files and labels
                for i in range(self.num_workers):
                    start_index = i * num_files // self.num_workers
                    end_index = (i + 1) * num_files // self.num_workers
                    index_list.append((start_index, end_index))
                # with Pool(self.num_workers) as p:
                with get_context("spawn").Pool(self.num_workers) as p:
                    ret = p.map(self.data_loader.get_valid_csi_data_files, index_list)
                    for valid_files, valid_labels, valid_track_ids, valid_receiver_ids, valid_user_ids in ret:
                        valid_files_list.extend(valid_files)
                        valid_labels_list.extend(valid_labels)
                        valid_track_ids_list.extend(valid_track_ids)
                        valid_receiver_ids_list.extend(valid_receiver_ids)
                        valid_user_ids_list.extend(valid_user_ids)
                    p.close()
                    p.join()
                self.file_paths = valid_files_list
                self.labels = valid_labels_list
                self.track_ids = valid_track_ids_list
                self.receiver_ids = valid_receiver_ids_list
                self.user_ids = valid_user_ids_list
                print("The number of valid CSI data after sanitizing: ", len(self.file_paths))
                
                valid_sample_list = {'file_paths': self.file_paths, 'labels': self.labels, 'track_ids': self.track_ids, 'receiver_ids': self.receiver_ids, 'user_ids': self.user_ids}
                with open(valid_sample_list_file_name, 'wb') as f:
                    pickle.dump(valid_sample_list, f)
                print("The valid file paths and labels are saved as ", valid_sample_list_file_name)
        print("Dataset initialization done.")
        
    def load_csi(self, file_name):
        if self.format == 'dense_dfs' or self.format == 'dense_dfs_amp': # reture the stored dfs (with stft.window_step = 10) data with shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
            dfs_file_name = file_name.replace('CSI', 'DFS')
            dfs_file_name = dfs_file_name.replace('dat', 'npy')
            dfs_data = np.load(dfs_file_name)
            time_bins = dfs_data.shape[0]
            if time_bins <= self.time_length // 10: # considering the time length of the dfs data under the window step of 10
                # padding the dfs data to the same length
                padded_dfs = np.zeros((self.time_length // 10, dfs_data.shape[1], dfs_data.shape[2], dfs_data.shape[3], dfs_data.shape[4]))
                padded_dfs[:time_bins, :, :, :, :] = dfs_data
            else:  # select the middle part of the dfs data
                start_index = (time_bins - self.time_length // 10) // 2
                padded_dfs = dfs_data[start_index:start_index+self.time_length // 10, :, :, :, :]
            if self.format == 'dense_dfs_amp':
                # the padded_dfs is the dfs data with shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2 (real and imag)]
                # the amplitude is the sqrt(real^2 + imag^2)
                dfs_amp = np.sqrt(padded_dfs[:, :, :, :, 0]**2 + padded_dfs[:, :, :, :, 1]**2)
                # add new axis to the amplitude data
                dfs_amp = np.expand_dims(dfs_amp, axis=-1) # shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 1]
                return torch.tensor(dfs_amp)
            return torch.tensor(padded_dfs)
        else:
            if 'CSI_Processed' in file_name:
                csi_matrix = np.load(file_name)
            else:
                csi_data = self.my_reader.read_file(file_name)  
                csi_matrix, _, _ = get_CSI(csi_data)  # shape: (frames, subcarriers, Rx, Tx)
                if self.preprocess:   # phase cleaning and filering
                    csi_matrix, _ = preprocess_CSI(csi_matrix)
            # padding the csi data to the same length
            num_frames, num_subcarriers, num_rx, num_tx = csi_matrix.shape
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
                        padded_csi = padded_csi[:self.time_length, :, :]
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
                freq_bin, ticks, freq_time_prof = get_csi_dfs(padded_csi, samp_rate = 1000, window_size = 256, window_step = 35) 
                # shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
                return freq_time_prof
            return padded_csi

    def parse_paths_labels(self):
        print("Parsing the file paths and labels...")
        inter_label_set = set(self.user_ids)
        self.label_mapping = {label: index for index, label in enumerate(inter_label_set)}
        print('The number of parsing files: ', len(self.file_names))
        for index, inter_f in enumerate(tqdm(self.file_names)):
            self.file_paths.append(os.path.join(self.config['dataset_path'], inter_f)) 
            self.labels.append(self.label_mapping[self.user_ids[index]])  
        print("Parsing done. The number of files: ", len(self.file_paths))     
           
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        if self.preload: # all the csi data is loaded into memory
            csi = self.data[idx]
        else: # using the lazy loading mode
            csi = self.load_csi(self.file_paths[idx])
        label = self.labels[idx]
        track_id = self.track_ids[idx]
        receiver_id = self.receiver_ids
        return csi, torch.tensor(label), torch.tensor(track_id), torch.tensor(receiver_id)
    
    def get_csi_device_info(self):
        return self.csi_device_info
    
    def get_label_name(self, label):
        inter_label = list(self.label_mapping.keys())[list(self.label_mapping.values()).index(label)]
        label_name = "user_" + str(inter_label)
        return label_name
    
    
# converting the data shape to the shape that the model needs
class widarGait_data_shape_converter:
    def __init__(self,config):
        self.model_input_shape = config['model_input_shape'] # BCHW, BLC, BTCHW, B2CNFT 
        self.config = config
        # B: batch size, C: channel, H: height, W: width, T: time length, N: number of (WiFi CSI) links, F: frequency bins
        self.format = config['format']
        try:
            self.z_score = config['z_score']
        except:
            self.z_score = False
        self.dense_dfs_virtual_phase = np.zeros(121)
        self.csi_virtual_phase = np.zeros(30)
        try:
            self.ppe = config['ppe'] # the flag of the physics prior embedding, None: no ppe, otherwise: the ppe is used
        except:
            self.ppe = None
    
    def shape_convert(self,batch):
        if self.format == 'dfs' or self.format == 'dense_dfs' or self.format == 'dense_dfs_amp':    # shape of the input batch: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2 or 1]
            try:
                self.align2widar3 = self.config['align2widar3']
            except:
                self.align2widar3 = False
            if self.align2widar3:
                # Efficiently align WidarGait data dimensions to Widar3 data shape
                # Align WidarGait data dimensions to Widar3 data shape
                # The main difference is in the time_bins dimension (WidarGait: 300, Widar3: 256)
                if batch.shape[1] > 256:
                    # If WidarGait time bins are more than 256, truncate to 256
                    batch = batch[:, :256, :, :, :, :]
                elif batch.shape[1] < 256:
                    # If WidarGait time bins are less than 256, pad with zeros to reach 256
                    padding_size = 256 - batch.shape[1]
                    padding = torch.zeros((batch.shape[0], padding_size, batch.shape[2], 
                                          batch.shape[3], batch.shape[4], batch.shape[5]), 
                                          device=batch.device, dtype=batch.dtype)
                    batch = torch.cat([batch, padding], dim=1)
                # Now batch has shape [batch_size, 256, freq_bins, num_subcarriers, Rx*Tx, 2 or 1]
                # which aligns with the Widar3 data format
            
            if self.z_score:
                # the input batch is the torch.tensor dfs data with shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
                # z score normalization except the the first dimension (batch_size) and the last dimension (real and imag)
                mean = batch.mean(dim=[1,2,3,4], keepdim=True)
                std = batch.std(dim=[1,2,3,4], keepdim=True)
                batch = (batch - mean) / std
            if self.model_input_shape == 'BCHW':
                # merge the last three dimensions of the batch into one dimension: num_subcarriers * Rx*Tx * 2
                if self.format == 'dense_dfs_amp':
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]*batch.shape[4]) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers * Rx*Tx]
                else:
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]*batch.shape[4]*2) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers * Rx*Tx * 2]
                # transpose the last dim to the second dim as channel
                batch = batch.permute(0,3,1,2)
            elif self.model_input_shape == 'BCHW-C': # the complex version of the BCHW
                if  self.format == 'dense_dfs_amp' and self.ppe is None:
                    # the batch now is the amplitude of the dfs data with shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx*(num_receivers), 1]
                    # convert to tensor and repeat to have the same shape as the amplitude
                    virtual_phase = torch.tensor(self.dense_dfs_virtual_phase).view(1, 1, self.dense_dfs_virtual_phase.shape[0], 1, 1, 1).repeat(batch.shape[0], batch.shape[1], 1, batch.shape[3], batch.shape[4], batch.shape[5])
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
                if self.format == 'dense_dfs_amp':
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]*batch.shape[4])
                else:
                    # merge the last four dimensions of the batch into one dimension: freq_bins*num_subcarriers * Rx*Tx * 2
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]*batch.shape[4]*2) # shape: [batch_size, Time_bins, freq_bins*num_subcarriers * Rx*Tx * 2]
            elif self.model_input_shape == 'BTCHW': # for baseline: Widar3.0 model
                if self.format == 'dense_dfs_amp':
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
                else:
                    # merge the last two dimensions of the batch into one dimension: Rx*Tx * 2
                    batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]*2) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx * 2]
                # transpose the last dim to the third dim as channel
                batch = batch.permute(0,1,4,2,3)  # shape: [batch_size, Time_bins, Rx*Tx * 2, freq_bins, num_subcarriers]
            elif self.model_input_shape == 'B2CNFT':  # for baseline: SLNet
                if self.format == 'dense_dfs_amp' and self.ppe is None:
                    # the batch now is the amplitude of the dfs data with shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx*(num_receivers), 1]
                    # convert to tensor and repeat to have the same shape as the amplitude
                    virtual_phase = torch.tensor(self.dense_dfs_virtual_phase).view(1, 1, self.dense_dfs_virtual_phase.shape[0], 1, 1, 1).repeat(batch.shape[0], batch.shape[1], 1, batch.shape[3], batch.shape[4], batch.shape[5])
                    real_part = batch*torch.cos(virtual_phase)
                    img_part = batch*torch.sin(virtual_phase)
                    batch = torch.cat((real_part, img_part), dim=-1) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx*(num_receivers), 2]
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
        return batch

def make_widar_gait_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd, num_workers = 0):
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

