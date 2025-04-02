import glob, os, math
from random import shuffle, seed
from CSIKit.reader import IWLBeamformReader
from torch.utils.data import Dataset, DataLoader
from .utils import get_CSI, preprocess_CSI, get_dfs, get_csi_dfs
from tqdm import tqdm
import torch
import yaml
import argparse
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import pickle
from einops import rearrange, repeat
# from multiprocessing import set_start_method
# set_start_method("spawn")
from multiprocessing import get_context
# We first prepare the meta information of the dataset and select the samples based on the input configuration.

USER_NUM = 17
ROOM_NUM = 3
GESTURE_NUM = 22

RAW_PREFIX = 'CSI'  # this is the folder where the raw data is stored
SORTED_GES_LIST = ['Slide', 'Push&Pull', 'Clap', 'Sweep', 'Draw-Zigzag(Horizontal)', 
                  'Draw-O(Horizontal)',
                  'Draw-N(Horizontal)', 'Draw-Rectangle(Horizontal)', 
                  'Draw-Triangle(Horizontal)', 'Draw-Zigzag(Vertical)', 'Draw-N(Vertical)', 
                  'Draw-4', 'Draw-5', 'Draw-7', 'Draw-3', 'Draw-2', 'Draw-9', 'Draw-1', 
                  'Draw-0', 'Draw-6', 'Draw-8', 'Draw-O(Vertical)'] 
# this is the list of gestures in the order of the labels we finnaly used. 

# thus the label of the gesture is the index of the gesture in the above list + 1
Gesture_index_name_map = {1: 'Slide', 2: 'Push&Pull', 3: 'Clap', 4: 'Sweep', 5: 'Draw-Zigzag(Horizontal)',
                          6: 'Draw-O(Horizontal)', 7: 'Draw-N(Horizontal)', 8: 'Draw-Rectangle(Horizontal)',
                          9: 'Draw-Triangle(Horizontal)', 10: 'Draw-Zigzag(Vertical)', 11: 'Draw-N(Vertical)',
                          12: 'Draw-4', 13: 'Draw-5', 14: 'Draw-7', 15: 'Draw-3', 16: 'Draw-2', 17: 'Draw-9',
                          18: 'Draw-1', 19: 'Draw-0', 20: 'Draw-6', 21: 'Draw-8', 22: 'Draw-O(Vertical)'
                          }

SKIP_LIST = [   
                "20181209/user6/user6-3-1-1-5",
                "20181211/user9/user9-3-4-2-3",
                "20181211/user9/user9-1-1-1-1",
                "20181211/user8/user8-3-3-3-5",
                "20181211/user8/user8-1-1-1-1",
                "20181211/user9/user9-4-5-5-2",
                "20181130/user12/user12-1-3-5-4",
                "20181130/user13/user13-6-1-3-2",
                "20181130/user14/user14-4-1-1-3",
                "20181130/user17/user17-5-5-2-2",
                "20181130/user17/user17-5-3-3-3",
                "20181130/user15/user15-3-5-1-2",
                "20181130/user15/user15-5-4-5-2",
            ]   # this skip list is from rfboost paper where they have mentioned that these files are not used in their experiments for resons like too few samples etc.


# the following class and function is used to generate the file list that satisfies the given conditions
class Widar3Meta:
    def __init__(self):
        self.file = ''
        self.room = 0
        self.gesture_map = {}
        self.gesture_list = []
        self.user = []
        self.sample_num = 0

    def parse_ges(self, ges_str):
        ges = ges_str.strip('; ').split(';')
        self.gesture_list = [g.strip() for g in ges if g != '']
        try:
            for g in self.gesture_list:
                idx = int(g.split(':')[0].strip())
                name = g.split(':')[1].strip()
                self.gesture_map[idx] = name
        except:
            print("error parsing gesture")
            
    def ges_tostring(self):
        return '; '.join(self.gesture_list) + ';'

    def sample_tostring(self):
        return '; '.join(str(s) for s in self.sample_num.values()) + ';'

    def user_tostring(self):
        return 'User' + ','.join(str(u) for u in self.user)

    def parse_user(self, user_str):
        self.user = [int(u) for u in user_str[4:].split(",")]

    def reader(self, line):
        self.file, self.room, ges_str, user_str, self.sample_num = line.strip().split('|')
        self.file = self.file.strip()
        self.room = int(self.room)
        self.sample_num = int(self.sample_num)
        self.parse_ges(ges_str)
        self.parse_user(user_str)

    def room_tostring(self):
        return '{}'.format(self.room)
    
    def make_filepattern_list(self, user_list=range(1, 1+USER_NUM), gesture_list=SORTED_GES_LIST, room_list = range(1,1+ROOM_NUM), ges_from_same_folder=False, as_list=False, ds_ratio=1, prefix="raw"):
        if user_list == []:
            user_list = range(1, 1+USER_NUM)
        if gesture_list == []:
            gesture_list = range(1,1+GESTURE_NUM)
        if room_list == []:
            room_list = range(1,1+ROOM_NUM)
        
        if ds_ratio != 1:
            as_list = True
        
        filelist = []
        filepattern_list = []
        total = 0
        user_sel, ges_sel, room_sel = [], [], []
        
        user_sel = set(user_list).intersection(set(self.user))
        room_sel = set(room_list).intersection(set([self.room]))
        if ges_from_same_folder:
            if set(list(gesture_list)).issubset(self.gesture_map.values()):
                ges_sel = gesture_list
        else:
            ges_sel = set(gesture_list).intersection(set(self.gesture_map.values()))
            
        # print(user_sel, room_sel, ges_sel)
        if len(user_sel) > 0 and len(room_sel) > 0 and len(ges_sel) > 0:
            for u in user_sel:
                for r in room_sel:
                    for g in ges_sel:
                        g_idx = list(self.gesture_map.keys())[list(self.gesture_map.values()).index(g)]
                        file_pattern = "{}/{}/{}/user{}/user{}-{}-*.dat".format(prefix, RAW_PREFIX, self.file, u, u, g_idx)
                        # print(file_pattern)
                        filepattern_list.append(file_pattern)
                        if as_list:
                            rawpattern = "{}/{}/{}/user{}/user{}-{}-*-r1.dat".format(prefix,RAW_PREFIX, self.file, u, u, g_idx)
                            # print('rawpattern: ', rawpattern)
                            files = glob.glob(rawpattern)
                            # files = [prefix + f.split(RAW_PREFIX+"/")[-1] for f in files]
                            files = [f.split("-r1.dat")[0] for f in files]
                            files = [f+" "+g for f in files]
                            n_files = int(len(files)*ds_ratio)
                            filelist.extend(files[:n_files])
                            total += n_files
                        else:
                            total += self.sample_num
        # else:
        #     print(user_list, gesture_list, room_list)
        #     print(self.user, self.room, self.gesture_map)
        #     raise ValueError("No valid user, room, gesture found in the meta data.")
        if as_list:
            return filelist, total
        else:
            return filepattern_list, total

    def header(self):
        return 'file|room|gesture_list|user|sample_num\n'
    def __str__(self):
        return '{}|{}|{}|{}|{}'.format(self.file, self.room_tostring(), self.ges_tostring(), self.user_tostring(), self.sample_num)


def files_lables(file_pattern_list, gesture_list=SORTED_GES_LIST, use_global_label=True):   
    file_paths = []
    labels = []

    for file_pattern in file_pattern_list:
        if file_pattern in SKIP_LIST:
            continue
        if not use_global_label:
            file_path = file_pattern.split(" ")[0]
            label = file_pattern.split("/")[-1].split("-")[1]
        else:
            gesture_map = {}
            for idx, name in enumerate(gesture_list):
                gesture_map[idx] = name
                
            file_path = file_pattern.split(" ")[0]
            g_str = file_pattern.split(" ")[1]
            label = list(gesture_map.keys())[list(gesture_map.values()).index(g_str)] + 1
        label = int(label)
        file_paths.append(file_path)
        labels.append(label)
    return file_paths, labels
        

def Widar3Database(config):
    print("Loading the Widar3 dataset...")
    print(config["dataset_path"])
    dataset_path = config["dataset_path"]
    user_list = config["user_list"]  # the selected user list
    gesture_slected = config["gesture_list"]       # the selected gesture list
    start_ges_index = gesture_slected[0]
    end_ges_index = gesture_slected[1]
    gesture_list = SORTED_GES_LIST[start_ges_index:end_ges_index]       # the selected gesture list
    ges_from_same_folder = config["ges_from_same_folder"]
    room_list=config["room_list"]    # the selected room list: 1, 2, 3
    
    # loading the meta information from the .csv file: The main reason is that the dataset file names are useing different index for the same gestures across different collecting time
    metas = []
    with open(os.path.join(dataset_path, "metas.csv")) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]    
        lines = lines[1:]
        for line in lines:   
            meta = Widar3Meta()
            meta.reader(line)
            metas.append(meta) 

    all_file_pattern = []
    for meta in metas:
        datalist, _ = meta.make_filepattern_list(user_list=user_list,
                                            prefix=dataset_path, 
                                            gesture_list=gesture_list, 
                                            room_list=room_list, 
                                            ges_from_same_folder=ges_from_same_folder, 
                                            as_list=True, 
                                            ds_ratio=1,
                                            )
        if datalist:
            all_file_pattern.extend(datalist)
    intermediate_file_paths, intermidiate_labels = files_lables(all_file_pattern, gesture_list=gesture_list, use_global_label=True)
    # here the intermediate_file_paths are the list of file paths (lack of the Rx part) and the intermidiate_labels are the corresponding gesture labels
    return intermediate_file_paths, intermidiate_labels


class multi_processing_data_loader:
    def __init__(self, file_name_list, config, label_list=None):
        self.file_name_list = file_name_list
        self.preprocess = config['preprocess']
        self.format = config['format']
        self.time_length = config['time_length']
        self.tqdm_disable = config['tqdm_disable']
        self.label_list = label_list
        self.preload = config['preload']
        self.data_path = config['dataset_path']

    def load_csi_data(self, indexes):  # loading the csi data from the file list
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
                    # print("Find none csi data.")
                    continue
                # skip the csi with short frames
                num_frames, num_subcarriers, num_rx, num_tx = csi_matrix.shape
                if num_frames < 1000:  # one second
                    csi_data_list.append(None)
                    # print("Find csi data with short frames.")
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
                elif self.format == 'amplitude':
                    csi = torch.abs(csi)
                    padded_csi = torch.zeros(num_frames, num_subcarriers, num_rx*num_tx)
                    padded_csi[:csi.shape[0], :csi.shape[1], :csi.shape[2]] = csi
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
                else:        
                    csi_data = my_reader.read_file(file_name)  
                    csi_matrix, _, _ = get_CSI(csi_data)
                    if csi_matrix is None:
                        continue
                    num_frames, _, _, _ = csi_matrix.shape
                    if num_frames < 1000:
                        continue
                    valid_files.append(file_name)
                    valid_labels.append(label_list[index])
        return valid_files, valid_labels
    
    def get_len(self):
        return len(self.file_name_list)
        
    
class Widar3_Dataset(Dataset):
    def __init__(self, intermediate_file_paths, intermidiate_labels, config):
        self.intermediate_file_paths = intermediate_file_paths
        self.intermidiate_labels = intermidiate_labels
        self.file_paths = []
        self.labels = []
        try:
            self.label_mapping = config['label_mapping']
        except:
            self.label_mapping = None
            
        try:
            self.augmentation_ratio = config['augmentation_ratio']
            print("The augmentation ratio is: ", self.augmentation_ratio)
        except:
            self.augmentation_ratio = 0

        self.select_rx = config['select_rx']
        self.parse_paths_labels()
        self.my_reader = IWLBeamformReader()
        self.num_workers = config['num_workers']
        self.csi_device_info = None
        self.preload = config['preload']
        self.tqdm_disable = config['tqdm_disable']
        self.preprocess = config['preprocess']
        self.batch_size = config['batch_size']
        self.format = config['format']
        self.time_length = config['time_length']
        self.dataset_path = config['dataset_path']
        try:
            self.concat_rx = config['concat_rx']    # the number of Rx to be concatenated
            self.concat_func = widar3_data_concatenator(config)
        except:
            self.concat_rx = False
        
        # get the csi device info
        print("Getting the CSI device info...")
        csi_data = self.my_reader.read_file(self.file_paths[0])
        if self.csi_device_info is None:
            self.csi_device_info = csi_data.get_metadata()

        if self.preload:  # preloading all the CSI data
            self.data_loader = multi_processing_data_loader(self.file_paths, config)
            print("Preloading the CSI data...")
            self.data = [None for i in range(len(self.file_paths))]
            num_files = self.data_loader.get_len()
            # split the data into num_workers parts
            index_list = []   # each element is a tuple (start_index, end_index)
            
            for i in range(self.num_workers):
                start_index = i * num_files // self.num_workers
                end_index = (i + 1) * num_files // self.num_workers
                index_list.append((start_index, end_index))
            print("start multi-processor loading")
            
            # with Pool(self.num_workers) as p:
            with get_context("spawn").Pool(self.num_workers) as p:
                ret = p.map(self.data_loader.load_csi_data, index_list)
                for csi_data, index_range in ret:
                    start_index, end_index = index_range
                    self.data[start_index:end_index] = csi_data
                p.close()
                p.join()
            print("Preloading done.")
            
            # fillter out the csi data with None
            none_values_index = [i for i, x in enumerate(self.data) if x is None]
            self.file_paths = [i for j, i in enumerate(self.file_paths) if j not in none_values_index]
            self.labels = [i for j, i in enumerate(self.labels) if j not in none_values_index]
            self.data = [i for j, i in enumerate(self.data) if j not in none_values_index]
            if self.concat_rx:  # concatenate the data from different receivers collected at the same time
                rx_index = config['select_rx']
                # Extract unique base file paths, i.e., the number of trails where each trail is a gesture performed by a user in a room that captured by all the receivers
                base_file_path_list = list(set([file_path.split("-r")[0] for file_path in self.file_paths]))
                # Generate tuples of matching file paths
                temp = [
                    tuple(p for p in (f"{base_path}-r{rx_id}.dat" for rx_id in rx_index) if p in self.file_paths)
                    for base_path in base_file_path_list
                    if all(f"{base_path}-r{rx_id}.dat" in self.file_paths for rx_id in rx_index)
                ]
                # Update file paths， labels， data
                self.labels = [self.labels[self.file_paths.index(item[0])] for item in temp]
                self.data = [self.concat_func.reveicer_concat([self.data[self.file_paths.index(item[i])] for i in range(len(rx_index))]) for item in temp]
                self.file_paths = temp
            print("The number of valid CSI data: ", len(self.data))
        else:
            print("Using the lazy loading mode.")
            print("The number of instances before data sanitization: ", len(self.file_paths))
            print("Sanitizing the CSI data...")
            
            # the valid sample list file name is based on the config file   
            user_list_str = ''.join(str(e) for e in sorted(config['user_list']))
            gesture_list_str = ''.join(str(e) for e in config['gesture_list'])
            room_list_str = ''.join(str(e) for e in config['room_list'])
            select_rx_str = ''.join(str(e) for e in config['select_rx'])
            valid_sample_list_file_name = config['dataset_name'] +  "_user" + user_list_str + "_gesture" + gesture_list_str + "_room" + room_list_str + "_rx" + select_rx_str + '_valid_file_label.pkl'
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
                
            if os.path.exists(valid_sample_list_file_name):  # loading the valid file paths and labels recording from the pickle file
                with open(valid_sample_list_file_name, 'rb') as f:
                    valid_file_label = pickle.load(f)
                    self.file_paths = valid_file_label['file_paths']
                    self.labels = valid_file_label['labels']
                    print("The valid file paths and labels are loaded from: ", valid_sample_list_file_name)
                    print("The number of valid CSI data after sanitizing: ", len(self.file_paths))
            else:
                self.data_loader = multi_processing_data_loader(self.file_paths, config, label_list=self.labels)  # only processing the file path, rather than load the data
                num_files = self.data_loader.get_len()
                # split the data into num_workers parts
                index_list = []
                valid_files_list = []
                valid_labels_list = []
                # using multi-processing to get the valid files and labels
                for i in range(self.num_workers):
                    start_index = i * num_files // self.num_workers
                    end_index = (i + 1) * num_files // self.num_workers
                    index_list.append((start_index, end_index))
                # with Pool(self.num_workers) as p:
                with get_context("spawn").Pool(self.num_workers) as p:
                    ret = p.map(self.data_loader.get_valid_csi_data_files, index_list)
                    for valid_files, valid_labels in ret:
                        valid_files_list.extend(valid_files)
                        valid_labels_list.extend(valid_labels)
                    p.close()
                    p.join()
                self.file_paths = valid_files_list
                self.labels = valid_labels_list
                print("The number of valid CSI data after sanitizing: ", len(self.file_paths))
                # save the valid file paths and labels as pickle files
                valid_file_label = {'file_paths': self.file_paths, 'labels': self.labels}
                with open(valid_sample_list_file_name, 'wb') as f:
                    pickle.dump(valid_file_label, f)
                print("The valid file paths and labels are saved as ", valid_sample_list_file_name)
            
            if self.concat_rx:
                print("Concatenating the CSI data...")
                rx_index = config['select_rx']
                # Extract unique base file paths
                base_file_path_list = list(set([file_path.split("-r")[0] for file_path in self.file_paths]))
                # Generate tuples of matching file paths
                if (self.format != 'dense_dfs' and self.format != 'dense_dfs_amp') and os.path.exists(self.dataset_path + '/CSI_Processed'):
                    temp = [
                        tuple(p for p in (f"{base_path}-r{rx_id}.npy" for rx_id in rx_index) if p in self.file_paths)
                        for base_path in base_file_path_list
                        if all(f"{base_path}-r{rx_id}.npy" in self.file_paths for rx_id in rx_index)
                    ]
                else: # we load the raw data from the 'CSI' folder
                    temp = [
                        tuple(p for p in (f"{base_path}-r{rx_id}.dat" for rx_id in rx_index) if p in self.file_paths)
                        for base_path in base_file_path_list
                        if all(f"{base_path}-r{rx_id}.dat" in self.file_paths for rx_id in rx_index)
                    ]
                # Update file paths， labels， data
                self.labels = [self.labels[self.file_paths.index(item[0])] for item in temp]
                self.file_paths = temp
                print("The number of valid CSI data after concatenating: ", len(self.file_paths))
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
            elif self.format == 'amplitude':  # shape: (frames, subcarriers, Rx*Tx)
                    csi = torch.abs(csi)
                    padded_csi = torch.zeros(num_frames, num_subcarriers, num_rx*num_tx)
                    padded_csi[:csi.shape[0], :csi.shape[1], :csi.shape[2]] = csi
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
        if self.label_mapping is None:
            print("Creating the label mapping...")
            inter_label_set = set(self.intermidiate_labels)
            self.label_mapping = {label: index for index, label in enumerate(inter_label_set)}
        else:
            print("Using the existing label mapping...")
        # get the real paths and mapping the labels to [0,1,..., len(gesture_list)-1] with a dictionary to record the mapping
        print('The number of parsing files: ', len(self.intermediate_file_paths))
        for index, inter_f in enumerate(tqdm(self.intermediate_file_paths)):
            # the file folder path is the part that before the last '/' in the inter_f
            file_folder_path = inter_f.rsplit('/', 1)[0]
            selected_file = inter_f.rsplit('/', 1)[1]
            # list all the files in the folder
            files = glob.glob(file_folder_path+"/*.dat")
            for file in files:
                if selected_file in file:
                    rx_id_temp = int(file.split("/")[-1].split("-")[5].split(".")[0][1:])
                    if rx_id_temp in self.select_rx:  # selected the samples with the selected Rx
                        self.file_paths.append(file)
                        self.labels.append(self.label_mapping[self.intermidiate_labels[index]])
        print("Parsing done. The total number of files = #_parsing_files x ~6 (six receivers): ", len(self.file_paths))
        
    def __len__(self):
        if self.preload:
            return len(self.data)
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        if self.concat_rx:
            file_path = self.file_paths[idx]  # the file path is a tuple of file paths
            if self.preload:
                csi = self.data[idx]
            else:
                csi = self.concat_func.reveicer_concat([self.load_csi(path) for path in file_path])
            label = self.labels[idx]
            file_path = file_path[0]
            user_id = int(file_path.split("/")[-1].split("-")[0][4:])
            orientation = int(file_path.split("/")[-1].split("-")[3])
            rx_id = 0 
        else:
            file_path = self.file_paths[idx] # the file path is a string
            if self.preload:
                csi = self.data[idx]
            else:
                csi = self.load_csi(file_path)
            label = self.labels[idx]
            user_id = int(file_path.split("/")[-1].split("-")[0][4:])
            orientation = int(file_path.split("/")[-1].split("-")[3])
            rx_id = int(file_path.split("/")[-1].split("-")[5].split(".")[0][1:])
        if self.augmentation_ratio > 0:
            # Apply data augmentation by adding random noise
            if isinstance(csi, np.ndarray):
                if np.iscomplexobj(csi):
                    # For complex data, add noise to the magnitude
                    magnitude = np.abs(csi)
                    phase = np.angle(csi)
                    mean_magnitude = np.mean(magnitude)
                    noise_level = mean_magnitude * self.augmentation_ratio
                    noise = np.random.normal(0, noise_level, magnitude.shape)
                    augmented_magnitude = magnitude + noise
                    # Convert back to complex
                    csi = augmented_magnitude * np.exp(1j * phase)
                else:
                    # For real data, add noise directly
                    mean_value = np.mean(np.abs(csi))
                    noise_level = mean_value * self.augmentation_ratio
                    noise = np.random.normal(0, noise_level, csi.shape)
                    csi = csi + noise
            elif isinstance(csi, torch.Tensor):
                if torch.is_complex(csi):
                    # For complex tensor, add noise to the magnitude
                    magnitude = torch.abs(csi)
                    phase = torch.angle(csi)
                    mean_magnitude = torch.mean(magnitude)
                    noise_level = mean_magnitude * self.augmentation_ratio
                    noise = torch.randn_like(magnitude) * noise_level
                    augmented_magnitude = magnitude + noise
                    # Convert back to complex
                    csi = augmented_magnitude * torch.exp(1j * phase)
                else:
                    # For real tensor, add noise directly
                    mean_value = torch.mean(torch.abs(csi))
                    noise_level = mean_value * self.augmentation_ratio
                    noise = torch.randn_like(csi) * noise_level
                    csi = csi + noise
        return csi, torch.tensor(label), torch.tensor(user_id), torch.tensor(orientation), torch.tensor(rx_id)
    
    def get_label_name(self, label):
        inter_label = list(self.label_mapping.keys())[list(self.label_mapping.values()).index(label)]
        label_name = Gesture_index_name_map[inter_label]
        return label_name
    
    def get_label_mapping(self):
        return self.label_mapping
    
    def get_csi_device_info(self):
        return self.csi_device_info
        # the meta_data is a CSIMetadata object, print all the attributes
            # print("chipset: ", meta_data.chipset)
            # print("backend: ", meta_data.backend)
            # print("bandwidth: ", meta_data.bandwidth)
            # print("antenna_config: ", meta_data.antenna_config)
            # print("frames: ", meta_data.frames)
            # print("subcarriers: ", meta_data.subcarriers)
            # print("time_length: ", meta_data.time_length)
            # print("average_sample_rate: ", meta_data.average_sample_rate)
            # print("average_rssi: ", meta_data.average_rssi)
            # print("csi_shape: ", meta_data.csi_shape)


def Get_Widar3_Dataset(config):
    intermediate_file_paths, intermidiate_labels = Widar3Database(config)
    dataset = Widar3_Dataset(intermediate_file_paths, intermidiate_labels, config)
    return dataset
    

# concatenate the csi data from different receivers
class widar3_data_concatenator:
    def __init__(self,config):
        self.format = config['format']
    
    def reveicer_concat(self,data_list):
        # stack the data from different receivers in the last and create a new dimension
        datas = torch.stack(data_list, dim=-1)  # shape: [*, num_receivers]
        if self.format == 'dfs' or self.format == 'dense_dfs' or self.format == 'dense_dfs_amp':    
            # mean = datas.mean(dim=[0,1,2,3,5], keepdim=True)
            # std = datas.std(dim=[0,1,2,3,5], keepdim=True)
            # datas = (datas - mean) / std
            # merge the Rx*Tx and num_receivers dimensions
            datas = rearrange(datas, 't f c x v r -> t f c (x r) v') # shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx*num_receivers, 2]
        elif self.format == 'polar' or self.format == 'cartesian' or self.format == 'amplitude': # [frames, subcarriers, Rx*Tx*2/1, num_receivers]
            # concatenate the tensor data from different receivers along the Rx*Tx*2 dimension
            # data = torch.cat(data_list, dim=2)  # shape: [frames, subcarriers, Rx*Tx*2]
            if self.format == 'amplitude':
                datas = rearrange(datas, 'f c (x d) r -> f c x d r', d=1) # shape: [frames, subcarriers, Rx*Tx, 1, num_receivers]
            else:
                datas = rearrange(datas, 'f c (x d) r -> f c x d r', d=2) # shape: [frames, subcarriers, Rx*Tx, 2, num_receivers]
            # mean = datas.mean(dim=[0,1,2], keepdim=True)
            # std = datas.std(dim=[0,1,2], keepdim=True)
            mean = datas.mean(dim=[0,1,2,4], keepdim=True)
            std = datas.std(dim=[0,1,2,4], keepdim=True)
            datas = (datas - mean) / std
            datas = rearrange(datas, 'f c x d r -> f c (x r d)') # shape: [frames, subcarriers, Rx*Tx*num_receivers*2]
        elif self.format == 'complex': # [frames, subcarriers, Rx*Tx, num_receivers]  in np.complex64]
            # # concatenate the tensor data from different receivers along the Rx*Tx dimension
            # data = torch.cat(data_list, dim=2)  # shape: [frames, subcarriers, Rx*Tx]
            # excute the z score normalization on the real and imaginary parts of the complex data
            real_part = datas.real # shape: [frames, subcarriers, Rx*Tx, num_receivers]
            img_part = datas.imag  # shape: [frames, subcarriers, Rx*Tx, num_receivers]
            # mean_real = real_part.mean(dim=[0,1,2], keepdim=True)
            # std_real = real_part.std(dim=[0,1,2], keepdim=True)
            # mean_img = img_part.mean(dim=[0,1,2], keepdim=True)
            # std_img = img_part.std(dim=[0,1,2], keepdim=True)
            mean_real = real_part.mean(dim=[0,1,2,3], keepdim=True)
            std_real = real_part.std(dim=[0,1,2,3], keepdim=True)
            mean_img = img_part.mean(dim=[0,1,2,3], keepdim=True)
            std_img = img_part.std(dim=[0,1,2,3], keepdim=True)
            real_part = (real_part - mean_real) / std_real
            img_part = (img_part - mean_img) / std_img
            datas = real_part + 1j*img_part
            datas = rearrange(datas, 'f c x r -> f c (x r)') # shape: [frames, subcarriers, Rx*Tx*num_receivers]
        else:
            raise ValueError("The config.format must be one of 'polar', 'cartesian', 'dfs', 'complex', 'amplitude', 'dense_dfs', 'dense_dfs_amp'.")
        # return data
        return datas
    

# converting the data shape to the shape that the model needs
class widar3_data_shape_converter:
    def __init__(self,config):
        self.model_input_shape = config['model_input_shape'] # BCHW, BLC, BTCHW, B2CNFT 
        # B: batch size, C: channel, H: height, W: width, T: time length, N: number of (WiFi CSI) links, F: frequency bins
        self.config = config
        self.format = config['format']
        try:
            self.z_score = config['z_score']
        except:
            self.z_score = False
        try:
            self.ppe = config['ppe'] # the flag of the physics prior embedding, None: no ppe, otherwise: the ppe is used
        except:
            self.ppe = None
        # we create a virtual phase for the dfs data similar to that in the SLNet paper
        # the virtual phase is from -pi to pi with 121 points (the number of dfs frequency bins)
        # self.dense_dfs_virtual_phase = np.linspace(-np.pi, np.pi, 121)
        # viral phase with zero phase
        self.dense_dfs_virtual_phase = np.zeros(121)
        # the virtual phase for the csi data is from -pi to pi with 30 points (the number of subcarriers)
        # self.csi_virtual_phase = np.linspace(-np.pi, np.pi, 30)
        # viral phase with zero phase
        self.csi_virtual_phase = np.zeros(30)
    
    def shape_convert(self,batch):
        if self.format == 'dfs' or self.format == 'dense_dfs' or self.format == 'dense_dfs_amp':    # shape of the input batch: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2 or 1]
            if self.z_score:
                # the input batch is the torch.tensor dfs data with shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
                # z score normalization except the the first dimension (batch_size) and the last dimension (real and imag)
                mean = batch.mean(dim=[1,2,3,4], keepdim=True)
                std = batch.std(dim=[1,2,3,4], keepdim=True)
                batch = (batch - mean) / std
            if self.model_input_shape == 'BCHW':
                # merge the last three dimensions of the batch into one dimension: num_subcarriers * Rx*Tx * 2
                batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]*batch.shape[4]*batch.shape[5]) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers * Rx*Tx * 2]
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
                # merge the last four dimensions of the batch into one dimension: freq_bins*num_subcarriers * Rx*Tx * 2
                batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]*batch.shape[4]*batch.shape[5]) # shape: [batch_size, Time_bins, freq_bins*num_subcarriers * Rx*Tx * 2]
            elif self.model_input_shape == 'BTCHW': # for baseline: Widar3.0 model
                # merge the last two dimensions of the batch into one dimension: Rx*Tx * 2
                batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]*batch.shape[5]) # shape: [batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx * 2]
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
        elif self.format == 'polar' or self.format == 'cartesian' or self.format == 'amplitude': 
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

def make_widar3_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd, num_workers=0):
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

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description="Code implementation")
    parser.add_argument("config_file", type=str, help="Configuration YAML file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    

    config['format'] = 'complex'   # 'polar', 'cartesian', 'complex', 'dfs'

    if len(config['data_split']) ==3:
        train_ratio = config['data_split'][0]
        val_ratio = config['data_split'][1]
        test_ratio = config['data_split'][2]
        merged_config = {**config, **config['all_dataset']} 
        dataset = Get_Widar3_Dataset(merged_config)
        # split the dataset with the given ratio: train_ratio, val_ratio, test_ratio
        train_num = int(len(dataset)*train_ratio)
        val_num = int(len(dataset)*val_ratio)
        test_num = int(len(dataset)) - train_num - val_num
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num], generator=rng_generator)
    else:
        train_set_config = config.copy().update(config['train_dataset'])
        train_set = Get_Widar3_Dataset(train_set_config)
        val_num = int(len(train_set)*(0.2))
        train_num = len(train_set) - val_num
        train_set, val_set = torch.utils.data.random_split(train_set, [train_num, val_num], generator=rng_generator)
        test_set_config = config.copy().update(config['test_dataset'])
        test_set = Get_Widar3_Dataset(test_set_config)
    
    data_sahpe_coverter = widar3_data_shape_converter(config)
    train_loader = make_widar3_dataloader(train_set, is_training=False, generator=rng_generator, batch_size=config['batch_size'],collate_fn_padd=None, num_workers=config['num_workers'])

    for i, sample in enumerate(train_loader):
        csi, label, user_id, orientation, rx_id = sample
        print('csi batch shape before converter: ',csi.shape)
        csi = data_sahpe_coverter.shape_convert(csi)
        print('csi batch shape after converter: ',csi.shape)
        
        print(sample[0].shape)
        print(sample[0][0,0,0,0]) 
        
        if config['format'] == 'polar':
            csi_amp = sample[0][0, :, :, 0]
            csi_phase = sample[0][0, :, :, 3]
            csi_complex = csi_amp * torch.exp(1j*csi_phase)
        elif config['format'] == 'cartesian':
            csi_real = sample[0][0, :, :, 0]
            csi_imag = sample[0][0, :, :, 3]
            csi_amp = torch.sqrt(csi_real**2 + csi_imag**2)
            csi_phase = torch.atan2(csi_imag, csi_real)
            csi_complex = csi_real + 1j*csi_imag
        elif config['format'] == 'complex':
            csi_amp = torch.abs(sample[0][0, :, :, 0])
            csi_phase = torch.angle(sample[0][0, :, :, 0])
            csi_complex = sample[0][0, :, :, 0]
        elif config['format'] == 'dfs':
            freq_time_prof = sample[0][0,:] # torch.Size([Batch_size, Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2])
            freq_time_prof_real = freq_time_prof[:,:,0,0,0]
            freq_time_prof_imag = freq_time_prof[:,:,0,0,1]
            doppler_spectrum = np.square(np.abs(freq_time_prof_real)) + np.square(np.abs(freq_time_prof_imag))
            doppler_spectrum = np.transpose(doppler_spectrum, (1, 0))
            # doppler_spectrum = np.square(np.abs(freq_time_prof))
            # doppler_spectrum = np.log10(np.square(np.abs(freq_time_prof)) + 1e-20) + 20
            doppler_phase = np.angle(freq_time_prof_real + 1j*freq_time_prof_imag)
            # doppler_phase = np.angle(freq_time_prof)
            plt.matshow(doppler_spectrum)
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()