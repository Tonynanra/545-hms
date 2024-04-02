# %%
import pandas as pd, numpy as np, sys
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed


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

def spectrogram_from_eeg(n):
    '''
    Returns: spectrogram in the shape (n=100, chns, f, t)
    '''
    raw_100 = np.load(DATAPATH+'train_eegs_processed/eeg_filt_{:04}.npy'.format(n+1))
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

    np.save(DATAPATH + 'custom_specs/spec_{:04}.npy'.format(n+1), spec_100)
    print(f'processed chunk {n}')
    f.flush()
    return spec_100.mean(), (spec_100 ** 2).mean()

# Function for parallel normalization and saving
def normalize_and_save(n):
    dat_100 = np.load(DATAPATH + 'custom_specs/spec_{:04}.npy'.format(n+1))
    dat_100 = (dat_100 - E_X) / np.sqrt(var)
    np.save(DATAPATH + 'custom_specs/spec_{:04}.npy'.format(n+1), dat_100.astype(np.float32))
    print(f"normalized chunk {n}")
    f.flush()
    return 0
    

#%%
if __name__ == '__main__':
    f = open('spec.log', 'w')
    sys.stdout = f
    sys.stderr = f
    np.seterr(all='raise')

    DATAPATH = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/hms_data/'
    RAW_DATAPATH = DATAPATH + 'raw_data/'

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

    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as executor:
        results = list(executor.map(spectrogram_from_eeg, range(1068)))
        for n, means in enumerate(results):
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