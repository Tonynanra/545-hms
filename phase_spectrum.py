# %%
import numpy as np
import sys
import cupy as cp
import pandas as pd
import cupyx.scipy.signal as sg
import concurrent.futures
import torch
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

#%%
def read_metadata():
    """
    Reads the training metadata from a CSV file and adds a column for the path to the corresponding EEG data.

    Returns:
        DataFrame: The training data with an added column for the EEG data path.
    """
    train_df = pd.read_csv(READ_BASEPATH+'train.csv')
    train_df = train_df.reset_index(drop=True) # reset the index to start from 0
    train_df['eeg_path'] = READ_BASEPATH+'train_eegs/'+train_df['eeg_id'].astype(str)+'.parquet'
    return train_df

def read_parquet_helper(i):
    """
    Reads the EEG data for a given ID from a Parquet file.

    Args:
        i (int): The ID of the EEG data to read.

    Returns:
        tuple: The ID and the EEG data as a DataFrame.
    """
    fn = READ_BASEPATH+'train_eegs/'+i.astype(str)+'.parquet'
    return i, pd.read_parquet(fn)

def get_eeg_data_slice(i, train_df, train_eegs):
    """
    Gets a slice of the EEG data for a given index.

    Args:
        i (int): The index of the EEG data to get.
        train_df (DataFrame): The training data.
        train_eegs (dict): The EEG data.

    Returns:
        ndarray: The slice of the EEG data.
    """
    start = (train_df['eeg_label_offset_seconds'][i]*200).astype(int)
    end = start + 200*50
    return train_eegs[train_df['eeg_id'][i]].iloc[start:end].to_numpy()

def process_nan(x):
    """
    Process NaN values in a multidimensional array along axis 1.

    Parameters:
    - x (ndarray): The input multidimensional array with NaN values.

    Returns:
    - (ndarray): The processed array with NaN values replaced by means and entirely NaN slices set to zero.
    """

    means = cp.nanmean(x, axis=1, keepdims=True)
    isnan = cp.isnan(x)
    all_nan_slice = isnan.all(axis=1, keepdims=True)
    all_nan_slice_expanded = cp.repeat(all_nan_slice, x.shape[1], axis=1)
    x = cp.where(isnan & ~all_nan_slice_expanded, means, x)
    x[all_nan_slice_expanded] = 0
    return x

def make_phase(dat_100):
    """
    Compute the phase spectrogram of the input data.

    Parameters:
    - dat_100 (ndarray): Input data in 100 batches of shape (100, fftax, chns=20).

    Returns:
    - f (ndarray): Array of frequencies.
    - t (ndarray): Array of time points.
    - phase (ndarray): Phase spectrogram of shape (100, chns=4, t, f).
    """
    f_s = 200
    l_s = 7
    dat_100 = cp.stack((dat_100[:,:,FEATS_idx[:,0]] - dat_100[:,:,FEATS_idx[:,1]],
                        dat_100[:,:,FEATS_idx[:,1]] - dat_100[:,:,FEATS_idx[:,2]],
                        dat_100[:,:,FEATS_idx[:,2]] - dat_100[:,:,FEATS_idx[:,3]],
                        dat_100[:,:,FEATS_idx[:,3]] - dat_100[:,:,FEATS_idx[:,4]]),axis=-1) #(100, fftax, chns=4, to_concat)
    dat_100 = process_nan(dat_100)
    f, t, phase = sg.spectrogram(dat_100, fs=f_s, nperseg=f_s*l_s, noverlap=f_s*l_s//1.03, window='boxcar', return_onesided=True, axis=1, mode='phase')
    phase = cp.moveaxis(phase, 1, -1).sum(axis=2) / 4 #(100, chns=4, t, f)
    phase = torch.as_tensor(phase, device='cuda')
    phase = torch.nn.functional.interpolate(phase, size=(256, 128), mode='bicubic', align_corners=False)
    return f, t, phase.cpu().numpy()

    


def normalize_phase(i, E_X, var):
    """
    Normalizes a single phase spectrum file to zero mean and unit variance.

    Parameters:
    - i (int): Index of the file to be processed
    - E_X (float): Mean of phases precalculated over the entire dataset.
    - var (float): Variance of phases precalculated over the entire dataset.
    """
    filepath = DATAPATH + 'phase_spectrum_raw/phase_{:04}.npy'.format(i+1)
    dat_100 = np.load(filepath)
    dat_100 = (dat_100 - E_X) / np.sqrt(var)
    np.save(filepath, dat_100.astype(np.float32))
    print(f"Processed file {i+1}")
    return 0

# %%
if __name__ == '__main__':
    f = open('phase.log', 'w')
    sys.stdout = f
    sys.stderr = f
    np.seterr(all='raise')
    DATAPATH = './'
    READ_BASEPATH = DATAPATH


    #Reading metadata
    train_df = read_metadata()

    #Reading EEG parquets
    unique_eeg_ids = train_df['eeg_id'].unique()
    train_eegs = dict()
    print("Reading EEG data")
    with ThreadPoolExecutor() as executor:
            for i, data in executor.map(read_parquet_helper, unique_eeg_ids):
                train_eegs[i] = data

    #Splitting EEG data
    eeg_channel_names = train_eegs[train_df['eeg_id'][0]].columns # needs to be relative to starting point
    print("Spliting EEG data")
    with ThreadPoolExecutor() as executor:
        train_final_data = list(executor.map(get_eeg_data_slice, range(len(train_df)), [train_df]*len(train_df), [train_eegs]*len(train_df)))
    del train_eegs

    NAMES = ['LL','LP','RP','RR']
    FEATS = [['Fp1','F7','T3','T5','O1'],
            ['Fp1','F3','C3','P3','O1'],
            ['Fp2','F8','T4','T6','O2'],
            ['Fp2','F4','C4','P4','O2']]
    FEATS_idx = np.array([[eeg_channel_names.get_loc(i) for i in j] for j in FEATS])

    E_X = 0
    E_Xsq = 0

    for i in np.arange(0, train_df.shape[0], 100):
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        torch.cuda.empty_cache()

        _, _, phase = make_phase(cp.array(np.stack(train_final_data[i:i+100])))
        E_X = (E_X * i + phase.mean()) / (i + 1)
        E_Xsq = (E_Xsq * i + (phase**2).mean()) / (i + 1)
        np.save(DATAPATH+'phase_spectrum_raw/phase_{:04}.npy'.format(i//100+1), phase.astype(np.float32))
    var = E_Xsq - E_X**2
    

    print("All Phases Generated\n\n")
    print(f"mean: {E_X}, var: {var}")
    print('*' * 25 + '\n\n')

    with ThreadPoolExecutor() as executor:
        # Submitting all tasks to the executor
        futures = [executor.submit(normalize_phase, i, E_X, var) for i in range(train_df.shape[0]//100)]
        
        # Optional: waiting for all futures to complete (for progress tracking or logging)
        for future in concurrent.futures.as_completed(futures):
            future.result()