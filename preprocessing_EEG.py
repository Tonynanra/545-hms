import pandas as pd
import numpy as np
import mne
import os
import traceback
import sys
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

#Set global variables
READ_BASEPATH = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/hms_data/raw_data/' # set read basepath
SAVE_BASEPATH = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/hms_data/' # set save basepath

SAVE_DIR = SAVE_BASEPATH+'train_eegs_processed/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

## set these params if need to resume preprocessing
# calculate by mult last saved npy chunk_num by 100, then subtract 100
# ie. eeg_filt_0087.npy => 87 * 100 = 8700 - 100 => start at 8600
START_IND = 47400

def read_metadata():
    """
    Reads the training metadata from a CSV file and adds a column for the path to the corresponding EEG data.

    Returns:
        DataFrame: The training data with an added column for the EEG data path.
    """
    train_df = pd.read_csv(READ_BASEPATH+'train.csv')[START_IND:] # index starts at START_IND
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
    ica = mne.preprocessing.ICA(n_components=10, random_state=97, verbose=40)
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


def save_eeg_data(eeg_filt):
    """
    Saves the preprocessed EEG data to disk.

    Args:
        eeg_filt (list): The preprocessed EEG data.
    """
    new_dir = SAVE_BASEPATH+'train_eegs_processed/'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    #EEG data is saved in chunks of 100 windows
    for i in range(0, len(eeg_filt), 100):
        chunk = eeg_filt[i:i+100]
        np.save(new_dir + f'eeg_filt_{i//100}', chunk)

def main():
    # f = open('output.log', 'w')
    # sys.stdout = f
    # sys.stderr = f
    mne.set_log_level('ERROR')

    #Reading metadata
    train_df = read_metadata()

    #Reading EEG parquets
    unique_eeg_ids = train_df['eeg_id'].unique()
    train_eegs = dict()
    train_df['eeg'] = ""
    print("Reading EEG data")
    with ThreadPoolExecutor() as executor:
            for i, data in executor.map(read_parquet_helper, unique_eeg_ids):
                train_eegs[i] = data

    #Splitting EEG data
    eeg_channel_names = train_eegs[train_df['eeg_id'][0]].columns # needs to be relative to starting point
    train_df['eeg'] = ""
    print("Spliting EEG data")
    with ThreadPoolExecutor() as executor:
        train_df['eeg'] = list(executor.map(get_eeg_data_slice, range(len(train_df)), [train_df]*len(train_df), [train_eegs]*len(train_df)))
    train_final_data = train_df['eeg']
    del train_eegs, train_df

    #Preprocessing EEG data
    x = ['eeg']*19
    x.append('ecg')
    info = mne.create_info(ch_names=eeg_channel_names.tolist(), sfreq=200,
                                        ch_types=x,
                                        verbose=False)
    eeg_filt = []
    exception_indices = []
    chunk_size = 100 # defines how often to save

    chunk_num = int(START_IND / 100) + 1 # used for saving
    
    # with ProcessPoolExecutor() as executor:
    #     futures = {executor.submit(preprocess_eeg_data_slice, train_final_data[i], info): i for i in range(0, len(train_final_data))}
    #     for future in concurrent.futures.as_completed(futures):
    #         i = futures[future]

    for i in range(len(train_final_data)):
        try:
            eeg_filt.append(preprocess_eeg_data_slice(train_final_data[i], info))
        except Exception as e:
            print(f"Processing failed for training window entry {i}")
            # print(traceback.format_exc())
            eeg_filt.append(train_final_data[i])
            exception_indices.append(i)
        
        # Save eeg_filt every 100 samples
        if (i + 1) % chunk_size == 0:
            # chunk_num = (i + 1) // chunk_size
            chunk_num_str = f"{chunk_num:04d}"
            # Save the NumPy array to a file
            np.save(f'{SAVE_DIR}eeg_filt_{chunk_num_str}.npy', np.array(eeg_filt))
            print(f'#### row {i} // eeg_filt_{chunk_num_str}.npy successfully saved ####')
            eeg_filt = []  # Reset eeg_filt for the next chunk
            chunk_num += 1

    # Save the remaining eeg_filt (if any) after the last chunk
    if eeg_filt:
        chunk_num_str = f"{chunk_num:04d}"
        np.save(f'{SAVE_DIR}eeg_filt_{chunk_num_str}.npy', np.array(eeg_filt))

    print(f"Processing failed for training window entries:\n{exception_indices}")
    

if __name__ == "__main__":
    main()
