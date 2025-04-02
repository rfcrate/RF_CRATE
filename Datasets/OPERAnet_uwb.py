import os
import numpy as np
import pandas as pd
import pickle   
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

activity_mapping = {
    'noactivity': 0, 
    'stand': 1,
    'sit': 2, 
    'liedown': 3, 
    'standfromlie': 4,
    'walk': 5, 
    'bodyrotate': 6,
}

person_id_mapping = {
    'One': 1,
    'Two': 2,
    'Three': 3,
    'Four': 4,
    'Five': 5,
    'Six': 6,
    'Seven': 7,
}

def dataloader(config):
    data_path = config['dataset_path'] # Open_Datasets/OPERAnet
    uwb_system_index = config['uwb_system_index'] # 1 or 2
    uwb_data_type = config['uwb_data_type'] # 'CIR' or 'CFR'
    if uwb_system_index != 1 and uwb_system_index != 2:
        raise ValueError('Invalid uwb_system_index')
    format = config['format'] # 'polar', 'cartesian', 'complex'
    tqdm_disable = config['tqdm_disable']
    time_length = config['time_length']
    
    Database = None
    if uwb_system_index == 1:
        with open(data_path +  '/OPERAnet_UWB1_Activity_Recognition_Database.pkl', 'rb') as f:
            Database = pickle.load(f)
    elif uwb_system_index == 2:
        with open(data_path + '/OPERAnet_UWB2_Activity_Recognition_Database.pkl', 'rb') as f:
            Database = pickle.load(f)

    person_id = Database['person_id']
    room_no = Database['room_no']
    activity = Database['activity']
    # fp_pow_dbm = Database['fp_pow_dbm']
    # fp_index = Database['fp_index']
    CIR = Database['CIR']
    
    data = []
    for i in range(len(CIR)):
        if uwb_data_type == 'CIR':
            data.append(CIR[i])
        elif uwb_data_type == 'CFR':
            data.append(np.fft.fft(CIR[i], axis=1))
        else:
            raise ValueError('Invalid uwb_data_type')

    samples = []
    labels = []
    pid = []
    rid = []
    for index in tqdm(range(len(data)), disable=tqdm_disable):
        data_temp = data[index]             # shape: T, 35/50  (UWB1: 35, UWB2: 50) :: [Slow time, Fast time]
        data_temp = np.array(data_temp, dtype=np.complex64)
        time_length_temp = data_temp.shape[0]
        
        if time_length_temp < time_length//3:  # if the time length is less than 1/3 of the required time length, skip the sample
            continue
        
        activity_label = activity_mapping[activity[index]]
        labels.append(activity_label)
        
        person_id_label = person_id_mapping[person_id[index]]
        pid.append(person_id_label)
        
        if time_length_temp < time_length:
            data_temp = np.concatenate((data_temp, np.zeros((time_length-time_length_temp, data_temp.shape[1]), dtype=np.complex64)), axis=0)
        elif time_length_temp > time_length:
            data_temp = data_temp[:time_length]
        data_temp = torch.tensor(data_temp, dtype=torch.complex64)
        if format == 'polar':
            amp = torch.abs(data_temp)
            data_temp = torch.stack((amp, torch.angle(data_temp)), dim=-1)  # shape: T, 35/50, 2
        elif format == 'cartesian':
            data_temp = torch.stack((data_temp.real, data_temp.imag), dim=-1)  # shape: T, 35/50, 2
        elif format == 'complex':
            pass
        else:
            raise ValueError('Invalid format')
        samples.append(data_temp.numpy())
        rid.append(room_no[index])
        
    samples = np.array(samples)
    labels = np.array(labels)
    pid = np.array(pid)
    rid = np.array(rid)
    return samples, labels, pid, rid


class OPERAnet_UWB_Dataset(Dataset):
    def __init__(self,config):
        self.config = config
        self.data, self.labels, self.person_id, self.room_no = dataloader(config)
        print("The number of samples in the dataset: ", len(self.data))
        self.format = config['format']
        selected_person_id = config['selected_user_id']
        selected_room_no = config['selected_room_no']
        # select the data from the selected person_id and room_no
        selected_indices = []
        for i in range(len(self.data)):
            if self.person_id[i] in selected_person_id and self.room_no[i] in selected_room_no:
                selected_indices.append(i)
        self.data = self.data[selected_indices]
        self.labels = self.labels[selected_indices]
        self.person_id = self.person_id[selected_indices]
        self.room_no = self.room_no[selected_indices]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        uid = self.person_id[idx]
        rid = self.room_no[idx]
        # convert the sample and label to tensor
        if self.format == 'complex':
            sample = torch.tensor(sample, dtype=torch.complex64)
        else:
            sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        uid = torch.tensor(uid, dtype=torch.long)
        rid = torch.tensor(rid, dtype=torch.long)
        return sample, label, uid, rid
    
    
# converting the data shape to the shape that the model needs
class OPERAnet_UWB_data_shape_converter:
    def __init__(self,config):
        self.model_input_shape = config['model_input_shape'] # BCHW, BLC, BTCHW, B2CNFT 
        # B: batch size, C: channel, H: height, W: width, T: time length, N: number of (WiFi CSI) links, F: frequency bins
        self.format = config['format']
    
    def shape_convert(self,batch):
        if self.format == 'polar' or self.format == 'cartesian':
        # shape of the input batch: [batch_size, slow_time, fast_time(35/50), amp_phase/real_img(2))
            if self.model_input_shape == 'BCHW':
                batch = batch.permute(0,3,1,2) 
            elif self.model_input_shape == 'BLC':
                batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]) 
            elif self.model_input_shape == 'BTCHW' or self.model_input_shape == 'B2CNFT':
                raise ValueError("The config.model_input_shape 'BTCHW' or 'B2CNFT' are not spported.") 
            else:
                raise ValueError("The config.model_input_shape must be one of 'BCHW', 'BLC'")
        elif self.format == 'complex':
         # shape of the input batch: [batch_size, slow_time, fast_time(35/50))
            if self.model_input_shape == 'BCHW':
                # add a new axis for the channel
                batch = batch.unsqueeze(1)
            elif self.model_input_shape == 'BTCHW' or self.model_input_shape == 'B2CNFT' or self.model_input_shape == 'BLC':
                raise ValueError("The config.model_input_shape 'BLC', 'BTCHW' or 'B2CNFT'  are not supported for config.format 'complex'.")
            else:
                raise ValueError("The config.model_input_shape must be 'BCHW'.")
        else:
            raise ValueError("The config.format must be one of 'polar', 'cartesian', 'complex'.")
        return batch

def make_OPERAnet_UWB_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd, num_workers = 0):
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

