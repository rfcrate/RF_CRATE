from CSIKit.reader import IWLBeamformReader
from CSIKit.tools.batch_graph import BatchGraph
import itertools
from typing import Tuple
from scipy.signal import butter, filtfilt
import numpy as np
import scipy.signal as signal
import torch

######################## IWL5300 CSI pre-processing tools based on the CSIKit library. ########################

def get_CSI(csi_data: 'CSIData', squeeze_output: bool = False) -> Tuple[np.array, int, int]:
    frames = csi_data.frames
    try:
        csi_shape = frames[0].csi_matrix.shape
    except:
        return (None,0,0)

    no_frames = len(frames)
    no_subcarriers = csi_shape[0]

    # Matrices should be Frames * Subcarriers * Rx * Tx.
    # Single Rx/Tx streams should be squeezed.
    if len(csi_shape) == 3:
        # Intel data comes as Subcarriers * Rx * Tx.
        no_rx_antennas = csi_shape[1]
        no_tx_antennas = csi_shape[2]
    elif len(csi_shape) == 2 or len(csi_shape) == 1:
        # Single antenna stream.
        no_rx_antennas = 1
        no_tx_antennas = 1
    else:
        # Error. Unknown CSI shape.
        print("Error: Unknown CSI shape.")

    csi = np.zeros((no_frames, no_subcarriers, no_rx_antennas, no_tx_antennas), dtype=complex)
    ranges = itertools.product(*[range(n) for n in [no_frames, no_subcarriers, no_rx_antennas, no_tx_antennas]])
    is_single_antenna = no_rx_antennas == 1 and no_tx_antennas == 1

    drop_indices = []
    for frame, subcarrier, rx_antenna_index, tx_antenna_index in ranges:
        frame_data = frames[frame].csi_matrix
        if subcarrier >= frame_data.shape[0]:
            # Inhomogenous component
            # Skip frame for now. Need a better method soon.
            continue

        subcarrier_data = frame_data[subcarrier]
        if subcarrier_data.shape != (no_rx_antennas, no_tx_antennas) and not is_single_antenna:
            if rx_antenna_index >= subcarrier_data.shape[0] or tx_antenna_index >= subcarrier_data.shape[1]:
                # Inhomogenous component
                # Skip frame for now. Need a better method soon.
                drop_indices.append(frame)
                continue
        csi[frame][subcarrier][rx_antenna_index][tx_antenna_index] = subcarrier_data if is_single_antenna else \
            subcarrier_data[rx_antenna_index][tx_antenna_index]  
    csi = np.delete(csi, drop_indices, 0)
    csi_data.timestamps = [x for i, x in enumerate(csi_data.timestamps) if i not in drop_indices]
    if squeeze_output:
        csi = np.squeeze(csi)
    return (csi, no_frames, no_subcarriers)

def nextpow2(i):
    return np.ceil(np.log2(i))
def autocorr(x):
    nFFT = int(2**(nextpow2(len(x))+1))
    F = np.fft.fft(x,nFFT)
    F = F*np.conj(F)
    acf = np.fft.ifft(F)
    acf = acf[0:len(x)] # Retain nonnegative lags
    acf = np.real(acf)
    acf = acf/(acf[0]+1e-7) # Normalize
    return acf

def preprocess_CSI(csi_matrix: np.array, csi_sample_rate: int = 1000):
    '''ArithmeticError
    Preprocesses the CSI matrix by applying phase cleaning (Widar3.0) and a lowpass and highpass filter.
    Then, calculates the motion statistics.
    The shape of the CSI matrix (complex-valued) should be Frames (time dimension) * Subcarriers * Rx * Tx.
    Returns the processed CSI matrix and the motion statistics.
    '''
    # define the lowpass and highpass filter
    half_rate = csi_sample_rate / 2
    uppe_orde = 6
    uppe_stop = 60
    if uppe_stop > half_rate:
        uppe_stop = half_rate - 1
    lowe_orde = 3
    lowe_stop = 2
    lu, ld = butter(uppe_orde, uppe_stop / half_rate, 'low')  
    hu, hd = butter(lowe_orde, lowe_stop / half_rate, 'high')
    
    num_frames, num_subcarriers, num_rx, num_tx = csi_matrix.shape
    # print("num_frames: ", num_frames)
    # print("num_subcarriers: ", num_subcarriers)
    # print("num_rx: ", num_rx)
    # print("num_tx: ", num_tx)
    # phase cleaning
    if num_rx == 1:
        cleaned_csi = csi_matrix
    elif num_rx == 2:
        cleaned_csi = np.zeros((num_frames, num_subcarriers, 1, num_tx), dtype=complex)
        cleaned_csi = csi_matrix[:, :, 0, :] * np.conj(csi_matrix[:, :, 1, :])
        cleaned_csi = np.expand_dims(cleaned_csi, axis=2)
    else:
        cleaned_csi = np.zeros((num_frames, num_subcarriers, num_rx, num_tx), dtype=complex)
        for i in range(num_rx):
            cleaned_csi[:, :, i, :] = csi_matrix[:, :, i, :] * np.conj(csi_matrix[:, :, (i + 1)%num_rx, :])

    # calculate the motion statistics
    num_frames, num_subcarriers, num_rx, num_tx = cleaned_csi.shape
    # print("csi shape")
    # for sliding window size = 500, step = 100,
    win_start = list(range(0, num_frames-100, 100))
    win_num = len(win_start)
    motion_statistics = np.zeros((win_num, num_subcarriers, num_rx, num_tx))
    for ti, w_start in enumerate(win_start):
        for j in range(num_rx):
            for k in range(num_tx):
                win_range = range(w_start, w_start+100)
                csi_jk = cleaned_csi[win_range, :, j, k].copy()
                csi_jk_amp = np.abs(csi_jk)
                # L2 norm across subcarriers
                time_len = csi_jk_amp.shape[0]
                csi_jk_norm = [csi_jk_amp[t, :] / np.linalg.norm(csi_jk_amp[t, :]) for t in range(time_len)]
                csi_jk_norm = np.array(csi_jk_norm)
                # print("csi_jk_norm shape: ", csi_jk_norm.shape)
                # Remove DC
                for i in range(num_subcarriers):
                    this_csi = csi_jk_norm[:, i]
                    # remove DC
                    this_csi = this_csi - np.mean(this_csi)
                    # 
                    # motion_statistics[i, j, k] = (np.dot(np.insert(this_csi, 0, 0), np.append(this_csi, 0))) / np.dot(this_csi, this_csi)
                    this_acf = autocorr(this_csi)
                    motion_statistics[ti, i, j, k] = this_acf[1]
                    
     # apply the lowpass filter
    for i in range(num_subcarriers):
        for j in range(num_rx):
            for k in range(num_tx):
                cleaned_csi[:, i, j, k] = filtfilt(lu, ld, cleaned_csi[:, i, j, k])
    # apply the highpass filter
    for i in range(num_subcarriers):
        for j in range(num_rx):
            for k in range(num_tx):
                cleaned_csi[:, i, j, k] = filtfilt(lu, ld, cleaned_csi[:, i, j, k])

    return cleaned_csi, motion_statistics

def get_csi_dfs(csi_data, samp_rate = 1000, window_size = 256, window_step = 30):
    '''
    input csi_data: [Time_dim, num_subcarriers, Rx*Tx,] (complex numpy array)
    return dfs spetrum with shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
    '''
    
    half_rate = samp_rate / 2
    uppe_stop = 60
    freq_bins_unwrap = np.concatenate((np.arange(0, half_rate, 1) / samp_rate, np.arange(-half_rate, 0, 1) / samp_rate))
    freq_lpf_sele = np.logical_and(np.less_equal(freq_bins_unwrap,(uppe_stop / samp_rate)),np.greater_equal(freq_bins_unwrap,(-uppe_stop / samp_rate)))
    freq_lpf_positive_max = 60
    
    # DC removal
    csi_data = csi_data.numpy()
    csi_data = csi_data - np.mean(csi_data, axis=0)
    noverlap = window_size - window_step
    
    freq, ticks, freq_time_prof_allfreq = signal.stft(csi_data, 
                                                      fs=samp_rate, 
                                                      nfft=samp_rate,
                                                        window=('gaussian', 
                                                                window_size), 
                                                        nperseg=window_size, 
                                                        noverlap=noverlap, 
                                                        return_onesided=False,
                                                        padded=True, 
                                                        axis=0)
    
    freq_time_prof_allfreq = np.array(freq_time_prof_allfreq)
    freq_time_prof = freq_time_prof_allfreq[freq_lpf_sele, :]  # shape: [freq_bins, num_subcarriers, Rx*Tx, Time_bins]
    freq_time_prof = np.roll(freq_time_prof, freq_lpf_positive_max, axis=0) # shape: [freq_bins, num_subcarriers, Rx*Tx, Time_bins]
    freq_bin = np.array(freq)[freq_lpf_sele]
    freq_bin = np.roll(freq_bin, freq_lpf_positive_max, axis=0)
    # change freq_time_prof shape to: [Time_bins, freq_bins, num_subcarriers, Rx*Tx]
    freq_time_prof = np.transpose(freq_time_prof, (3, 0, 1, 2))
    freq_time_prof_real = freq_time_prof.real
    freq_time_prof_imag = freq_time_prof.imag
    # we get a tensor with shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
    freq_time_prof = np.stack([freq_time_prof_real, freq_time_prof_imag], axis=-1)
    return freq_bin, ticks, torch.tensor(freq_time_prof)


def get_dfs(csi_data, samp_rate = 1000, window_size = 256, window_step = 30):
    '''
    input csi_data: [Batch_size, Time_dim, num_subcarriers, Rx*Tx,] (complex torch tensor)
    return dfs spetrum with shape: [Batch_size, 2, num_subcarriers, Rx*Tx, freq_bins, Time_bins]
    samp_rate = 1000, window_size = 256, window_step = 10
    '''
    
    # transpose the csi_data to [Time_dim, Batch_size, num_subcarriers, Rx*Tx]
    csi_data = csi_data.permute(1,0,2,3)
    # print(csi_data.shape)
    
    half_rate = samp_rate / 2
    uppe_stop = 60
    freq_bins_unwrap = np.concatenate((np.arange(0, half_rate, 1) / samp_rate, np.arange(-half_rate, 0, 1) / samp_rate))
    freq_lpf_sele = np.logical_and(np.less_equal(freq_bins_unwrap,(uppe_stop / samp_rate)),np.greater_equal(freq_bins_unwrap,(-uppe_stop / samp_rate)))
    freq_lpf_positive_max = 60
    
    # DC removal
    csi_data = csi_data.numpy()
    csi_data = csi_data - np.mean(csi_data, axis=0)
    noverlap = window_size - window_step
    
    freq, ticks, freq_time_prof_allfreq = signal.stft(csi_data, 
                                                      fs=samp_rate, 
                                                      nfft=samp_rate,
                                                        window=('gaussian', 
                                                                window_size), 
                                                        nperseg=window_size, 
                                                        noverlap=noverlap, 
                                                        return_onesided=False,
                                                        padded=True, 
                                                        axis=0)
    
    # freq_time_prof_allfreq = np.array(freq_time_prof_allfreq)
    freq_time_prof = freq_time_prof_allfreq[freq_lpf_sele, :]
    freq_bin = np.array(freq)[freq_lpf_sele]
    freq_time_prof = np.roll(freq_time_prof, freq_lpf_positive_max, axis=0) # shape: [freq_bins, Batch_size, num_subcarriers, Rx*Tx, Time_bins]
    freq_bin = np.roll(freq_bin, freq_lpf_positive_max, axis=0)
    freq_time_prof = torch.tensor(freq_time_prof).permute(1,2,3,0,4) # shape: [Batch_size, num_subcarriers, Rx*Tx, freq_bins, Time_bins]
    freq_time_prof_real = freq_time_prof.real
    freq_time_prof_imag = freq_time_prof.imag
    # stack the real and imaginary part along the first dimension, then we get a tensor with shape: [Batch_size, 2, num_subcarriers, Rx*Tx, freq_bins, Time_bins]
    freq_time_prof = torch.stack([freq_time_prof_real, freq_time_prof_imag], dim=1)

    return freq_bin, ticks, freq_time_prof
