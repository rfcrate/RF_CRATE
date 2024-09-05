import os
import numpy as np
import pandas as pd
import pickle   
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


def dataloader(config):
    data_path = config['dataset_path'] # Open_Datasets/HuPR
    file_indexes = config['file_indexes'] # the indexes of the files to be loaded
    if len(file_indexes) == 0:  # load all the files from 1 to 235
        file_indexes = list(range(1, 236))
    format = config['format'] # 'polar', 'cartesian', 'complex'
    tqdm_disable = config['tqdm_disable']
    # PAth to the preprocessed radar maps and labels
    path_to_data = data_path + '/radar_maps/'
    files = os.listdir(path_to_data)
    # each file contains 600 data samples

    radar_maps = []
    joints_list = []
    bbox_list = []
    image_paths = []

    for f in tqdm(files, disable=tqdm_disable):
        file_id = int(f.split('_')[1].split('.')[0])
        if file_id in file_indexes:
            data = pickle.load(open(path_to_data + f, 'rb'))
            horizontal_radar_map = data['hori']  # List of maps with shape: (64, 64, 8) (range_bin, azimuth_bin, elevation_bin), complex values
            vertical_radar_map = data['vert']   # List of maps with shape: (64, 64, 8), complex values
            labels = data['labels']
            rgb_image_folder = data['rgb_image_folder']
            path_to_rgb_images = data_path + '/' + rgb_image_folder
            
            # for i in tqdm(range(len(horizontal_radar_map))):
            # we will take the sample every 10th frame: since the radar maps are captured at 10 fps, 
            # we can reduce the redundancy by taking every 10th frame.
            for i in range(0, len(horizontal_radar_map), 5):
                h_map = horizontal_radar_map[i]  # complex values, shape: (64, 64, 8)
                v_map = vertical_radar_map[i]   # complex values, shape: (64, 64, 8)
                label = labels[i]
                joints = label['joints']
                bbox = label['bbox']
                path_to_rgb_image = path_to_rgb_images + '/%09d.jpg'%(int(i))
                # stack the horizontal and vertical radar maps
                radar_map = np.stack((h_map, v_map), axis=-1)  # shape: (64, 64, 8, 2)
                radar_map = torch.tensor(radar_map, dtype=torch.complex64) # shape: (64, 64, 8, 2)

                if format == 'polar':
                    radar_map_amp = torch.abs(radar_map)
                    radar_map = torch.stack((radar_map_amp, torch.angle(radar_map)), dim=-1)  # shape: (64, 64, 8, 2, 2)
                elif format == 'cartesian':
                    radar_map = torch.stack((radar_map.real, radar_map.imag), dim=-1) # shape: (64, 64, 8, 2, 2)
                elif format == 'complex':
                    pass
                else:
                    raise ValueError('Invalid format')
                radar_maps.append(radar_map)
                image_paths.append(path_to_rgb_image)
                joints_list.append(joints)
                bbox_list.append(bbox)
    # convert the list to tensor
    radar_maps = torch.stack(radar_maps, dim=0)  # shape: (num_samples, 64, 64, 8, 2, 2) or (num_samples, 64, 64, 8, 2) for complex format
    joints = torch.tensor(joints_list)  # shape: (num_samples, 21, 3)
    bbox = torch.tensor(bbox_list)  # shape: (num_samples, 4)
    return radar_maps, joints, bbox, image_paths


class HuPR_Dataset(Dataset):
    def __init__(self,config):
        self.config = config
        self.radar_maps, self.joints, self.bbox, self.image_path = dataloader(config)
        print("The number of samples in the dataset: ", len(self.radar_maps))

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        radar_map = self.radar_maps[idx]
        label = self.joints[idx]
        bbox = self.bbox[idx]
        # image_path = self.image_path[idx]
        return radar_map, label, bbox
    
    def get_image_path(self, idx):
        return self.image_path[idx]
    
    
# converting the data shape to the shape that the model needs
class HuPR_data_shape_converter:
    def __init__(self,config):
        self.model_input_shape = config['model_input_shape'] # BCHW, BLC, BTCHW, B2CNFT 
        # B: batch size, C: channel, H: height, W: width, T: time length, N: number of (WiFi CSI) links, F: frequency bins
        self.format = config['format']
    
    def shape_convert(self,batch):
        if self.format == 'polar' or self.format == 'cartesian':
        # shape of the input batch: [batch_size, range_bin(64), azimuth_bin(64), elevation_bin(8), vertical+horizontal_radar(2), amp_phase/real_img(2))
            if self.model_input_shape == 'BCHW':
                # merge the last three dimensions to one dimension
                batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]*batch.shape[4]*batch.shape[5]) # shape: [batch_size, range_bin, azimuth_bin, elevation_bin * 2 * 2]
                batch = batch.permute(0,3,1,2) # shape: [batch_size, elevation_bin * 2 * 2, range_bin, azimuth_bin]
            elif self.model_input_shape == 'BLC':
                batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2]*batch.shape[3]*batch.shape[4]*batch.shape[5]) # shape: [batch_size, range_bin, azimuth_bin * elevation_bin * 2 * 2]
            elif self.model_input_shape == 'BLC-2': # this is for stfnet, and units model
                batch = batch.view(batch.shape[0], batch.shape[1]*batch.shape[2], batch.shape[3]*batch.shape[4]*batch.shape[5]) # shape: [batch_size, range_bin * azimuth_bin, elevation_bin * 2 * 2]
            elif self.model_input_shape == 'BTCHW':
                batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]*batch.shape[5]) # shape: [batch_size, range_bin, azimuth_bin, elevation_bin, 2 * 2]
                batch = batch.permute(0,1,4,2,3) # shape: [batch_size, range_bin, 2 * 2, azimuth_bin, elevation_bin]
            elif self.model_input_shape == 'B2CNFT':
                batch = batch.permute(0,5,3,4,2,1) # shape: [batch_size, amp_phase/real_img(2), elevation_bin(8),vertical+horizontal_radar(2), azimuth_bin, range_bin]
            else:
                raise ValueError("The config.model_input_shape must be one of 'BCHW', 'BLC', 'BTCHW', 'B2CNFT'.")
        elif self.format == 'complex':
        # shape of the input batch: [batch_size, range_bin(64), azimuth_bin(64), elevation_bin(8), vertical+horizontal_radar(2))
            if self.model_input_shape == 'BCHW':
                batch = batch.view(batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]*batch.shape[4]) # shape: [batch_size, range_bin, azimuth_bin, elevation_bin * 2]
                batch = batch.permute(0,3,1,2)
            elif self.model_input_shape == 'BTCHW' or self.model_input_shape == 'B2CNFT' or self.model_input_shape == 'BLC':
                raise ValueError("The config.model_input_shape 'BLC', 'BTCHW' or 'B2CNFT'  only supported by config.format 'polar' and 'cartesian'.")
            else:
                raise ValueError("The config.model_input_shape must be 'BCHW'.")
        else:
            raise ValueError("The config.format must be one of 'polar', 'cartesian', 'complex'.")
        return batch

def make_HuPR_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd, num_workers = 0):
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

