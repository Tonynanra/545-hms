import pandas as pd
import mne

def read_metaData(train_csv_path, test_csv_path):
    """
    Reads the train and test CSV files and adds the path of the EEG and spectrogram data to the dataframes.

    Args:
    train_csv_path (str): The file path of the train CSV file.
    test_csv_path (str): The file path of the test CSV file.

    Returns:
    tuple: Tuples containing the train dataframe and the test dataframe.
    """
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Add the path of the EEG and spectrogram data to the dataframes
    train_df['eeg_path'] = f'train_eegs/'+train_df['eeg_id'].astype(str)+'.parquet'
    train_df['spec_path'] = f'train_spectrograms/'+train_df['spectrogram_id'].astype(str)+'.parquet'

    test_df['eeg_path'] = f'test_eegs/'+test_df['eeg_id'].astype(str)+'.parquet'
    test_df['spec_path'] = f'test_spectrograms/'+test_df['spectrogram_id'].astype(str)+'.parquet'

    return train_df, test_df

def read_parquet(train_df):
    """
    Reads EEG data from parquet files and adds it to the train dataframe.

    Args:
        train_df (pandas.DataFrame): The train dataframe containing EEG data.

    Returns:
        tuple: A tuple containing the modified train dataframe and the EEG channel names.
    """
    # Get the unique EEG IDs
    unique_eeg_ids = train_df['eeg_id'].unique()
    train_eegs = dict()
    train_df['eeg'] = ""

    # Read the EEG data for each unique EEG ID and add it to the train dataframe
    for i in unique_eeg_ids:
        fn = f'train_eegs/'+i.astype(str)+'.parquet'
        train_eegs[i] = pd.read_parquet(fn)

    # Get the EEG channel names
    eeg_channel_names = train_eegs[train_df['eeg_id'][0]].columns

    # Add the EEG data to the train dataframe
    for i in range(len(train_df)):
        start = (train_df['eeg_label_offset_seconds'][i]*200).astype(int)
        end = start + 200*50
        train_df['eeg'][i] = train_eegs[train_df['eeg_id'][i]].iloc[start:end].to_numpy()

    # Drop unnecessary columns from the train dataframe
    train_final_data = train_df.drop(['eeg_id', 'eeg_sub_id', 'eeg_label_offset_seconds', 'spectrogram_id', 'spectrogram_sub_id', 'spectrogram_label_offset_seconds', 'label_id', 'patient_id', 'eeg_path', 'spec_path'], axis = 1)

    return train_final_data, eeg_channel_names

def preprocess_eeg(data, eeg_channel_names):
    """
    Preprocesses the EEG data.

    Args:
        data (dict): Dictionary containing the EEG data.
        eeg_channel_names (numpy.ndarray): Array of EEG channel names.

    Returns:
        dict: Dictionary containing the preprocessed EEG data.
    """
    x = ['eeg']*19
    x.append('ecg')

    # Create the info object for the raw data
    info = mne.create_info(ch_names=eeg_channel_names.tolist(), sfreq=200,
                                        ch_types=x,
                                        verbose=False)

    data['eeg_filt'] = ""

    # Preprocess the EEG data for each row in the train dataframe
    for i in range(len(data['eeg'])):
        Xnp = data['eeg'][i]

        raw = mne.io.RawArray(data=Xnp.T, info=info, verbose=False)
        raw.set_montage('standard_1005', on_missing='warn')

        raw_filt = raw.copy().filter(0.5,80,verbose=False)

        ssp_projectors = raw_filt.info["projs"]
        raw_filt.del_proj()

        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_filt)

        ica = mne.preprocessing.ICA(n_components=10, random_state=97)
        ica.fit(raw_filt)

        ecg_indices, ecg_scores = ica.find_bads_ecg(raw_filt)
        ica.exclude.extend(ecg_indices)

        raw_ica = ica.apply(raw_filt)

        raw_final_filt = raw_ica.notch_filter(freqs=50, filter_length='auto', phase='zero')
        raw_final_filt = raw_final_filt.notch_filter(freqs=60, filter_length='auto', phase='zero')

        data['eeg_filt'][i] = raw_final_filt.get_data()

    return data

def main():
    train_df, test_df = read_metaData(f"train.csv", f"test.csv")
    train_final_data, eeg_channel_names = read_parquet(train_df)
    train_final_data = preprocess_eeg(train_final_data, eeg_channel_names)

if __name__ == "__main__":
    main()