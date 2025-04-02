import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle
from torch.utils.data import DataLoader

class RPI_Dataset(Dataset):
    """
    Dataset for loading CSI data and corresponding respiration rates.
    
    Args:
        csi_dir: Directory containing CSI numpy files
        resp_rate_path: Path to the respiration rate pickle file
        time_length: Length of each segment in seconds
        time_step: Step size between segments in seconds
        csi_sampling_rate: Sampling rate of the CSI data in Hz
        resp_avg_window: Number of seconds at the end of each segment to average for ground truth
    """
    
    def __init__(self, config):
        self.csi_dir = config['csi_dir']
        self.resp_rate_path = config['resp_rate_path']
        self.time_length = config['seg_time_length']
        self.time_step = config['seg_time_length']
        self.csi_sampling_rate = config['csi_sampling_rate']
        self.resp_avg_window = config['resp_avg_window']
        self.format = config['format']
        self.selected_chunk_index = config['selected_chunk_index']  #[1,2,3,4,6,7,8,9,10] 
        
        # Load respiration rate data
        with open(self.resp_rate_path, 'rb') as f:
            resp_data = pickle.load(f)
        self.resp_rates = resp_data['resp_rates']
        self.resp_times = resp_data['times']
        
        # Calculate number of frames per segment and step
        self.frames_per_segment = int(self.time_length * self.csi_sampling_rate)
        self.step_frames = int(self.time_step * self.csi_sampling_rate)
        
        # Load CSI chunks and prepare segment indices
        self.segments = []
        self._load_csi_chunks()
        
    def _load_csi_chunks(self):
        """Load CSI chunks and prepare segment indices."""
        # Get all available chunk files
        chunk_files = [f for f in os.listdir(self.csi_dir) if f.startswith('chunk') and f.endswith('.npy')]
        chunk_files.sort(key=lambda x: int(x.replace('chunk', '').replace('.npy', '')))
        
        # Process each chunk file
        for chunk_idx, chunk_file in enumerate(chunk_files):
            chunk_id = int(chunk_file.replace('chunk', '').replace('.npy', ''))
            chunk_path = os.path.join(self.csi_dir, chunk_file)
            csi_data = np.load(chunk_path)
            if chunk_id not in self.selected_chunk_index:
                continue
            
            # Calculate how many segments we can extract from this chunk
            num_frames = csi_data.shape[0]
            num_segments = (num_frames - self.frames_per_segment) // self.step_frames + 1
            
            # leave the last segment
            num_segments = num_segments - 1
            
            # Store segment information
            for seg_idx in range(num_segments):
                start_frame = seg_idx * self.step_frames
                end_frame = start_frame + self.frames_per_segment
                
                if end_frame <= num_frames:
                    # Store chunk index, start frame, and end frame
                    self.segments.append({
                        'chunk_idx': chunk_id,  # +1 because filenames start with chunk1, chunk2, etc.
                        'start_frame': start_frame,
                        'end_frame': end_frame
                    })
                    
    
    def __len__(self):
        """Return the number of segments in the dataset."""
        return len(self.segments)
    
    def __getitem__(self, idx):
        """
        Get a segment of CSI data and its corresponding respiration rate.
        
        Args:
            idx: Segment index
            
        Returns:
            tuple: (csi_segment, resp_rate)
        """
        segment_info = self.segments[idx]
        chunk_id = segment_info['chunk_idx']
        start_frame = segment_info['start_frame']
        end_frame = segment_info['end_frame']
        # Load the CSI data for this chunk
        chunk_path = os.path.join(self.csi_dir, f'chunk{chunk_id}.npy')
        csi_data = np.load(chunk_path)
        # Extract the segment
        csi_segment = csi_data[start_frame:end_frame]
        csi_tensor = torch.from_numpy(csi_segment).cfloat()   # shape: time, subcarrier, rx*tx, 1 (complex) [1980, 114, 1, 1]
        csi_tensor = torch.reshape(csi_tensor, (csi_tensor.shape[0], csi_tensor.shape[1], csi_tensor.shape[2]*csi_tensor.shape[3]))
        
        if self.format == 'polar':
            csi_tensor_amp = np.abs(csi_tensor)
            csi_tensor_phase = np.angle(csi_tensor)
            csi_tensor = np.concatenate((csi_tensor_amp, csi_tensor_phase), axis=-1) # shape: [Time_length, num_subcarriers, Rx*Tx*2]
        elif self.format == 'cartesian':
            csi_tensor_real = np.real(csi_tensor)
            csi_tensor_imag = np.imag(csi_tensor)
            csi_tensor = np.concatenate((csi_tensor_real, csi_tensor_imag), axis=-1) # shape: [Time_length, num_subcarriers, Rx*Tx*2]
            csi_tensor = np.abs(csi_tensor)    # shape: [Time_length, num_subcarriers, Rx*Tx]
        elif self.format == 'amplitude':
            csi_tensor = np.abs(csi_tensor)    # shape: [Time_length, num_subcarriers, Rx*Tx]
        elif self.format == 'complex': # shape: [Time_length, num_subcarriers, Rx*Tx] in complex64
            pass

        # Determine the corresponding respiration rate
        # Each chunk corresponds to one respiration rate sequence
        resp_rate_sequence = self.resp_rates[chunk_id - 1]  # -1 because resp_rates is 0-indexed
        # Calculate the time position in the respiration sequence
        # Given that respiration is measured every second (601 values, as mentioned)
        segment_end_time = end_frame / self.csi_sampling_rate  # Time in seconds
        resp_end_idx = min(int(segment_end_time), len(resp_rate_sequence) - 1)
        resp_start_idx = max(0, resp_end_idx - self.resp_avg_window)
        # Ensure we have at least one value
        if resp_start_idx == resp_end_idx:
            resp_start_idx = max(0, resp_end_idx - 1)
        # Get the average of the last seconds of respiration rate
        avg_resp_rate = np.mean(resp_rate_sequence[resp_start_idx:resp_end_idx+1])  # Include the end index
        avg_resp_rate = np.array([avg_resp_rate])
        avg_resp_rate = torch.tensor(avg_resp_rate, dtype=torch.float32)
        
        return csi_tensor, avg_resp_rate



# converting the data shape to the shape that the model needs
class RPI_data_shape_converter:
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
        if self.format == 'polar' or self.format == 'cartesian' or self.format == 'amplitude': 
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

def make_RPI_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd, num_workers = 0):
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



# Example usage:
if __name__ == "__main__":
    config = {}
    config['csi_dir'] = '../Open_Datasets/RPI-AX200/Processed_data/csi'
    config['resp_rate_path'] = '../Open_Datasets/RPI-AX200/Processed_data/respiration_rate.pkl'
    config['seg_time_length'] = 10  # 10 seconds per segment
    config['seg_time_length'] = 2     # 5 seconds step between segments
    config['csi_sampling_rate'] = 198  # Hz
    config['resp_avg_window'] = 2  # Use last 3 seconds for respiration rate averaging
    config['format'] = 'polar'
    config['selected_chunk_index'] = [1,2,3,4,]
    config['model_input_shape'] = 'BCHW'
    # Create dataset
    dataset = RPI_Dataset(config)
    
    print(f"Dataset contains {len(dataset)} segments")
    csi_segment, resp_rate = dataset[0]
    print(f"CSI segment shape: {csi_segment.shape}")
    print(f"CSI segment type: {csi_segment.dtype}")
    print(f"Respiration rate: {resp_rate.item()} BPM")
    # Create dataloader
    dataloader = make_RPI_dataloader(dataset, is_training=True, generator=None, batch_size=1, collate_fn_padd=None, num_workers=0)
    converter = RPI_data_shape_converter(config)
    for batch in dataloader:
        csi_segment, resp_rate = batch
        csi_segment = converter.shape_convert(csi_segment)
        print(f"CSI segment shape: {csi_segment.shape}")
        print(f"CSI segment type: {csi_segment.dtype}")
        print(f"Respiration rate: {resp_rate.item()} BPM")
        break