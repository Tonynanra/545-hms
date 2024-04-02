# %%
import numpy as np
import sys
import cupy as cp
import pandas as pd
import cupyx.scipy.signal as sg
import concurrent.futures
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

#%%
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
    f, t, phase = sg.spectrogram(dat_100, fs=f_s, nperseg=f_s*l_s, noverlap=f_s*l_s//1.01, window='boxcar', return_onesided=True, axis=1, mode='phase')
    phase = cp.moveaxis(phase, 1, -1).sum(axis=2) / 4 #(100, chns=4, t, f)
    return f, t, phase.get()

    


def normalize_phase(i, E_X, var):
    """
    Normalizes a single phase spectrum file to zero mean and unit variance.

    Parameters:
    - i (int): Index of the file to be processed
    - E_X (float): Mean of phases precalculated over the entire dataset.
    - var (float): Variance of phases precalculated over the entire dataset.
    """
    filepath = DATAPATH + 'phase_spectrum/phase_{:04}.npy'.format(i+1)
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
    RAW_DATAPATH = DATAPATH


    train_df = pd.read_csv(RAW_DATAPATH + 'train.csv')
    unique_eeg_ids = train_df['eeg_id'].unique()[:2]
    train_eegs = dict()
    train_df['eeg'] = ""
    for i in unique_eeg_ids:
        fn = f'{RAW_DATAPATH}train_eegs/'+i.astype(str)+'.parquet'
        train_eegs[i] = pd.read_parquet(fn)
    eeg_channel_names = train_eegs[train_df['eeg_id'][0]].columns
    NAMES = ['LL','LP','RP','RR']

    FEATS = [['Fp1','F7','T3','T5','O1'],
            ['Fp1','F3','C3','P3','O1'],
            ['Fp2','F8','T4','T6','O2'],
            ['Fp2','F4','C4','P4','O2']]
    FEATS_idx = np.array([[eeg_channel_names.get_loc(i) for i in j] for j in FEATS])
    E_X = 0
    E_Xsq = 0

    for i in range(train_df.shape[0]//100):
        cp._default_memory_pool.free_all_blocks()
        dat_100 = cp.load(DATAPATH+'train_eegs_processed/eeg_filt_{:04}.npy'.format(i+1))
        _, _, phase = make_phase(dat_100)
        E_X = (E_X * i + phase.mean()) / (i + 1)
        E_Xsq = (E_Xsq * i + (phase**2).mean()) / (i + 1)
        np.save(DATAPATH+'phase_spectrum/phase_{:04}.npy'.format(i+1), phase)
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