# %%
import pandas as pd
import mne

# %% [markdown]
# ## Reading training and testing csv and parquet files
train_df = pd.read_csv(f'train.csv')
test_df = pd.read_csv(f'test.csv')


## Parquet File Names
train_df['eeg_path'] = f'train_eegs/'+train_df['eeg_id'].astype(str)+'.parquet'
train_df['spec_path'] = f'train_spectrograms/'+train_df['spectrogram_id'].astype(str)+'.parquet'

test_df['eeg_path'] = f'test_eegs/'+test_df['eeg_id'].astype(str)+'.parquet'
test_df['spec_path'] = f'test_spectrograms/'+test_df['spectrogram_id'].astype(str)+'.parquet'


# ## Adding corresponding 50s eeg and 10min spectogram to each row of training dataframe
unique_eeg_ids = train_df['eeg_id'].unique()
unique_spec_ids = train_df['spectrogram_id'].unique()

# reading all eegs
train_eegs = dict()
train_df['eeg'] = ""

for i in unique_eeg_ids:
    fn = f'train_eegs/'+i.astype(str)+'.parquet'
    train_eegs[i] = pd.read_parquet(fn)

eeg_channel_names = train_eegs[train_df['eeg_id'][0]].columns


# %% getting 50s eeg data for each row
for i in range(len(train_df)):
    start = (train_df['eeg_label_offset_seconds'][i]*200).astype(int)
    end = start + 200*50
    train_df['eeg'][i] = train_eegs[train_df['eeg_id'][i]].iloc[start:end].to_numpy()

train_final_data = train_df.drop(['eeg_id', 'eeg_sub_id', 'eeg_label_offset_seconds', 'spectrogram_id', 'spectrogram_sub_id', 'spectrogram_label_offset_seconds', 'label_id', 'patient_id', 'eeg_path', 'spec_path'], axis = 1)

#%% Preprocessing 


## EEG Info for Preprocessing

x = ['eeg']*19
x.append('ecg')

info = mne.create_info(ch_names=eeg_channel_names.tolist(), sfreq=200,
                                        ch_types=x,
                                        verbose=False)

#%%
train_final_data['eeg_filt'] = ""

# %%

for i in range(2): #range(len(train_final_data['eeg']))
    Xnp = train_final_data['eeg'][i]

    raw = mne.io.RawArray(data=Xnp.T, info=info, verbose=False)
    raw.set_montage('standard_1005', on_missing='warn')

    # Low-pass and High-pass Filters
    raw_filt = raw.copy().filter(0.5,80,verbose=False)

    ## TODO: Check this (not doing anything)
    ssp_projectors = raw_filt.info["projs"]
    raw_filt.del_proj()

    # ECG Summary Plot
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_filt)

    # ICA
    ica = mne.preprocessing.ICA(n_components=10, random_state=97); #change n_components
    ica.fit(raw_filt)

    # For ECG artifacts
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw_filt); #, ecg_events=ecg_events)
    ica.exclude.extend(ecg_indices)

    raw_ica = ica.apply(raw_filt)

    # Notch Filters
    raw_final_filt = raw_ica.notch_filter(freqs=50, filter_length='auto', phase='zero')
    raw_final_filt = raw_final_filt.notch_filter(freqs=60, filter_length='auto', phase='zero')

    # saving the preprocessed eeg data
    train_final_data['eeg_filt'][i] = raw_final_filt.get_data()