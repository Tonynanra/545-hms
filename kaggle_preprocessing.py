### follow subsetting in kaggle, apply our own preprocessing and save as npy file


import pandas as pd
import numpy as np
import mne
import os
import mne
from glob import glob
from tqdm import tqdm

READ_BASEPATH = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/hms_data/raw_data/' # set read basepath
SAVE_BASEPATH = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/hms_data/' # set save basepath

OUTPUT_DIR = "/kaggle/working/"
TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
TRAIN_EEGS = READ_BASEPATH+'train_eegs/'

SAVE_DIR = SAVE_BASEPATH+'all_filt_eegs/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def eeg_from_parquet_ecg_only(parquet_path: str, display: bool = False) -> np.ndarray:
    """
    This function reads a parquet file and extracts the middle 50 seconds of readings. Then it fills NaN values
    with the mean value (ignoring NaNs).
    :param parquet_path: path to parquet file.
    :param display: whether to display EEG plots or not.
    :return data: np.array of shape  (time_steps, eeg_features) -> (10_000, 8)
    """
    
    ecg_feature =  ['EKG']
    
    # === Extract middle 50 seconds ===
    eeg = pd.read_parquet(parquet_path, columns=ecg_feature)
    rows = len(eeg)
    offset = (rows - 10_000) // 2 # 50 * 200 = 10_000
    eeg = eeg.iloc[offset:offset+10_000] # middle 50 seconds, has the same amount of readings to left and right
    if display: 
        plt.figure(figsize=(10,5))
        offset = 0
    # === Convert to numpy ===
    data = np.zeros((10_000, 1)) # create placeholder of same shape with zeros
    for index, feature in enumerate(ecg_feature):
        x = eeg[feature].values.astype('float32') # convert to float32
        mean = np.nanmean(x) # arithmetic mean along the specified axis, ignoring NaNs
        nan_percentage = np.isnan(x).mean() # percentage of NaN values in feature
        # === Fill nan values ===
        if nan_percentage < 1: # if some values are nan, but not all
            x = np.nan_to_num(x, nan=mean)
        else: # if all values are nan
            x[:] = 0
        data[:, index] = x
        if display: 
            if index != 0:
                offset += x.max()
            plt.plot(range(10_000), x-offset, label=feature)
            offset -= x.min()
    if display:
        plt.legend()
        name = parquet_path.split('/')[-1].split('.')[0]
        plt.yticks([])
        plt.title(f'EEG {name}',size=16)
        plt.show()    
    return data

def append_ecg_feature(all_eegs, train_df):
    ## need to read through data and append ecg channel for our own preprocessing method
    all_eegs_copy = all_eegs
    eeg_paths = glob(TRAIN_EEGS + "*.parquet")
    eeg_ids = train_df.eeg_id.unique()

    for i, eeg_id in tqdm(enumerate(eeg_ids)):  
        # Save EEG to Python dictionary of numpy arrays
        eeg_path = TRAIN_EEGS + str(eeg_id) + ".parquet"
        data = eeg_from_parquet_ecg_only(eeg_path, display=False)              
        all_eegs_copy[eeg_id] = np.hstack((all_eegs[eeg_id], data))
    return all_eegs_copy


def preprocess_eeg_data_slice(eeg_data, info):
    """
    Preprocesses a slice of the EEG data.

    Args:
        eeg_data (ndarray): The slice of the EEG data to preprocess.
        info (Info): The MNE Info object containing the EEG channel names.

    Returns:
        ndarray: The preprocessed EEG data.
    """
    # Assign the input EEG data to Xnp
    Xnp = eeg_data

    # Create a RawArray object from the EEG data
    raw = mne.io.RawArray(data=Xnp.T, info=info, verbose=False)
    # Set the montage to 'standard_1005'
    raw.set_montage('standard_1005', on_missing='ignore')

    # Filter the raw data to keep frequencies between 0.5 and 80 Hz
    raw_filt = raw.copy().filter(0.5,80,verbose=False)

    # Delete projection items from the filtered data
    raw_filt.del_proj()

    # Initialize an ICA object with 10 components and a random state of 97
    ica = mne.preprocessing.ICA(n_components=None, random_state=97, verbose=False)
    # Fit the ICA to the filtered data
    ica.fit(raw_filt)

    # Find ECG artifacts in the filtered data
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw_filt)
    # Exclude the ECG artifacts from the ICA
    ica.exclude.extend(ecg_indices)

    # Apply the ICA to the filtered data
    raw_ica = ica.apply(raw_filt)

    # Apply a notch filter at 50 Hz and 60 Hz to the ICA data
    raw_final_filt = raw_ica.notch_filter(freqs=50, filter_length='auto', phase='zero')
    raw_final_filt = raw_final_filt.notch_filter(freqs=60, filter_length='auto', phase='zero')

    # Return the preprocessed EEG data
    return raw_final_filt.get_data().T



def main():

    train_df = pd.read_csv(READ_BASEPATH+'train.csv')

    ecg_feature =  ['EKG']
    eeg_features = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']

    # load the kaggle dataset created
    all_eegs = np.load(SAVE_BASEPATH+'eegs.npy',allow_pickle=True).item()

    ## preprocess the eeg data
    all_filt_eegs = dict()
    eeg_channel_names = eeg_features + ecg_feature # this is using the selected 8 features + EKG
    
    all_eegs_copy = append_ecg_feature(all_eegs, train_df)

    #Preprocessing EEG data
    x = ['eeg']*8
    x.append('ecg')
    info = mne.create_info(ch_names=eeg_channel_names, sfreq=200,
                                            ch_types=x,
                                            verbose=False)

    start_count = 0
    chunk_size = 100 # defines how often to save
    chunk_num = int(start_count / 100) + 1 # used for saving

    for k in list(all_eegs.keys())[start_count:]:
        unfilt_eeg = all_eegs_copy[k]
        try:
            filt_eeg = preprocess_eeg_data_slice(unfilt_eeg, info)
        except Exception as e:
            print(f"Processing failed for eeg_id {k}")
            filt_eeg = all_eegs_copy[k]
        all_filt_eegs[k] = filt_eeg
        
        if (start_count+1) % chunk_size == 0:
            chunk_num_str = f"{chunk_num:03d}"
            # Save the NumPy array to a file
            np.save(f'{SAVE_DIR}all_filt_eegs{chunk_num_str}.npy', all_filt_eegs)
            print(f'#### last start_count: {start_count} // all_filt_eegs{chunk_num_str}.npy successfully saved ####')
            all_filt_eegs = dict()  # Reset eeg_filt for the next chunk
            chunk_num += 1
        
        start_count += 1

    # Save the remaining eeg_filt (if any) after the last chunk
    if all_filt_eegs:
        chunk_num_str = f"{chunk_num:03d}"
        np.save(f'{SAVE_DIR}all_filt_eegs{chunk_num_str}.npy', all_filt_eegs)
    
    # np.save(SAVE_BASEPATH+'all_filt_eegs', all_filt_eegs)

if __name__ == "__main__":
    main()