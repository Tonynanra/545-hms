# %%
import pandas as pd, numpy as np, sys
import librosa
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

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

    means = np.nanmean(x, axis=1, keepdims=True)
    isnan = np.isnan(x)
    all_nan_slice = isnan.all(axis=1, keepdims=True)
    all_nan_slice_expanded = np.repeat(all_nan_slice, x.shape[1], axis=1)
    x = np.where(isnan & ~all_nan_slice_expanded, means, x)
    x[all_nan_slice_expanded] = 0
    return x

def spectrogram_from_eeg(raw_100, n):
    '''
    Returns: spectrogram in the shape (n=100, chns, f, t)
    '''

    dat_100 = np.stack((raw_100[:,:,FEATS_idx[:,0]] - raw_100[:,:,FEATS_idx[:,1]],
                        raw_100[:,:,FEATS_idx[:,1]] - raw_100[:,:,FEATS_idx[:,2]],
                        raw_100[:,:,FEATS_idx[:,2]] - raw_100[:,:,FEATS_idx[:,3]],
                        raw_100[:,:,FEATS_idx[:,3]] - raw_100[:,:,FEATS_idx[:,4]]),axis=-1)
    dat_100 = process_nan(dat_100) 
    dat_100 = np.moveaxis(dat_100, 1, -1) #(100, chns=4, to_concat, fftax)

    spec_100 = librosa.feature.melspectrogram(y=dat_100, sr=200, hop_length=dat_100.shape[-1]//256, 
                    n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

    width = (spec_100.shape[-1]//32)*32
    spec_100 = spec_100[...,:width]
    for i in range(spec_100.shape[0]):
        for j in range(spec_100.shape[1]):
            for k in range(spec_100.shape[2]):
                spec_100[i,j,k] = librosa.power_to_db(spec_100[i,j,k], ref=np.max)
    spec_100 = spec_100.sum(axis=2) / 4

    np.save(DATAPATH + 'custom_specs_raw/spec_{:04}.npy'.format(n+1), spec_100)
    print(f'processed chunk {n}')
    f.flush()
    return spec_100.mean(), (spec_100 ** 2).mean()

# Function for parallel normalization and saving
def normalize_and_save(n):
    dat_100 = np.load(DATAPATH + 'custom_specs_raw/spec_{:04}.npy'.format(n+1))
    dat_100 = (dat_100 - E_X) / np.sqrt(var)
    np.save(DATAPATH + 'custom_specs_raw/spec_{:04}.npy'.format(n+1), dat_100.astype(np.float32))
    print(f"normalized chunk {n}")
    f.flush()
    return 0
    

#%%
if __name__ == '__main__':
    # f = open('spec.log', 'w')
    # sys.stdout = f
    # sys.stderr = f
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

    with ProcessPoolExecutor() as executor:
        futures = list(executor.map(spectrogram_from_eeg, [np.stack(train_final_data[i:i+100]) for i in np.arange(0, train_df.shape[0], 100)], range(1068)))
        for n, means in enumerate(futures):
            E_X = (E_X * n + means[0]) / (n+1)
            E_Xsq = (E_Xsq * n + means[1]) / (n+1)

    var = E_Xsq - E_X**2
    print(f'\n\nMean: {E_X}\tVariance: {var}\n\n')

    # Second loop: Normalize and save in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(normalize_and_save, i) for i in range(1068)]
        for future in as_completed(futures):
            future.result()

    # for n in range(1068):
    #     means = spectrogram_from_eeg(n)
    #     E_X = (E_X * n + means[0]) / (n+1)
    #     E_Xsq = (E_Xsq * n + means[1]) / (n+1)

    # var = E_Xsq - E_X**2

    # for n in range(1068):
    #     dat_100 = np.load(DATAPATH + 'custom_specs/spec_{:04}.npy'.format(n+1))
    #     dat_100 = (dat_100 - E_X) / np.sqrt(var)
    #     np.save(DATAPATH + 'custom_specs/spec_{:04}.npy'.format(n+1), dat_100.astype(float32))