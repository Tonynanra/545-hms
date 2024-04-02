# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> HMS: <span style='color:#F1A424'>Harmful Brain Activity Classification</span><span style='color:#ABABAB'> [Train]</span></b> 
# 
# ***
# 
# **Consider upvoting this notebook if you find it useful 🙌🏼**
# 
# - [Inference Notebook](https://www.kaggle.com/alejopaullier/hms-efficientnetb0-pytorch-inference)
# - In case you don't want to train the model you can find my dataset with [5-fold trained Efficientnet models](https://www.kaggle.com/datasets/alejopaullier/hms-efficientnetb0-5-folds) that are the result of running this notebook.
# 
# This is the **PyTorch 🔥 version** of [Chris Deotte EfficientNetB0 Starter](https://www.kaggle.com/code/cdeotte/efficientnetb0-starter-lb-0-43#Train-DataLoader) give him an upvote too ⬆️!
# 
# Your goal in this competition is to detect and classify seizures and other types of harmful brain activity. You will develop a model trained on electroencephalography (EEG) signals recorded from critically ill hospital patients.
# 
# In this notebook you will learn how to train a `efficientnet` model for image classification using PyTorch. Hope you enjoy it and find it useful.
# 
# ### <b><span style='color:#F1A424'>Table of Contents</span></b> <a class='anchor' id='top'></a>
# <div style=" background-color:#3b3745; padding: 13px 13px; border-radius: 8px; color: white">
# <li> <a href="#introduction">Introduction</a></li>
# <li> <a href="#install_libraries">Install libraries</a></li>
# <li><a href="#import_libraries">Import Libraries</a></li>
# <li><a href="#configuration">Configuration</a></li>
# <li><a href="#utils">Utils</a></li>
# <li><a href="#load_data">Load Data</a></li>
# <li><a href="#preprocessing">Data Pre-processing</a></li>
# <li><a href="#validation">Validation</a></li>
# <li><a href="#dataset">Dataset</a></li>
# <li><a href="#dataloader">DataLoader</a></li>
# <li><a href="#model">Model</a></li>
# <li><a href="#scheduler">Scheduler</a></li>
# <li><a href="#loss">Loss Function</a></li>
# <li><a href="#functions">Train and Validation Functions</a></li>
# <li><a href="#train_loop">Train Loop</a></li>
# <li><a href="#train_full">Full Train</a></li>
# <li><a href="#train">Train</a></li>
# </div>
# 
# 
# # <b><span style='color:#F1A424'>|</span> Introduction</b><a class='anchor' id='introduction'></a> [↑](#top) 
# 
# ***
# 
# ### <b><span style='color:#F1A424'>What is an EEG waveform?</span></b>
# 
# **EEG** (Electroencephalogram) waveforms are the **patterns of electrical activity generated by the brain**, which are recorded using electrodes placed on the scalp. EEG is a non-invasive method that measures the electrical potentials produced by the firing of neurons in the brain. These electrical potentials are then amplified and displayed as waveforms on a computer or paper.
# 
# - **Delta Waves (0.5-4 Hz):** Delta waves are slow-wave patterns associated with deep sleep and certain abnormal brain states. They are usually the dominant waves during deep sleep stages.
# - **Theta Waves (4-8 Hz):** Theta waves are associated with drowsiness, relaxation, and the early stages of sleep. They can also be present during deep meditation.
# - **Alpha Waves (8-13 Hz):** Alpha waves are dominant when a person is awake but relaxed and not actively processing information. They are commonly seen when a person's eyes are closed.
# - **Beta Waves (13-30 Hz):** Beta waves are associated with active, alert, and focused mental activity. They are commonly observed when a person is awake and engaged in cognitive tasks.
# - **Gamma Waves (30-100 Hz and above):** Gamma waves are associated with higher cognitive functions, such as perception, learning, and problem-solving. They are not always present and are often associated with specific cognitive tasks.
# 
# In this competition, EEG waveforms are 50 seconds long.
# 
# ### <b><span style='color:#F1A424'>What is a spectogram?</span></b>
# 
# A spectrogram is a visual representation of the spectrum of frequencies in a signal as they vary with time. It is a three-dimensional plot that displays how the frequencies of a signal change over time. Spectrograms are commonly used in signal processing, audio analysis, and other fields to analyze the frequency content of a signal and how it evolves over time.
# 
# ### <b><span style='color:#F1A424'>Useful References</span></b>
# 
# - [Understand this competition's data](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010)
# 

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Import Libraries</b><a class='anchor' id='import_libraries'></a> [↑](#top) 
# 
# ***
# 
# Import all the required libraries for this notebook.

# %%
import gc
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import random
import time
import timm
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import warnings
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Configuration</b><a class='anchor' id='configuration'></a> [↑](#top) 
# 
# ***

# %%
class config_debug:
    AMP = True
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VALID = 32
    EPOCHS = 4
    FOLDS = 5
    FREEZE = False
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1e7
    MODEL = "tf_efficientnet_b0"
    NUM_FROZEN_LAYERS = 39
    USE_CUSTOM_SPECS = True
    USE_PHASE_SPECS = True
    OVERLAPPING_SPECS = False
    NUM_WORKERS = 0
    PRINT_FREQ = 20
    SEED = 545
    TRAIN_FULL_DATA = False
    K_FOLD_EXIST = True
    TRAIN_SHUFFLE = False
    VISUALIZE = True
    WEIGHT_DECAY = 0.01

class config_train:
    AMP = True
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VALID = 32
    EPOCHS = 5
    FOLDS = 5
    FREEZE = False
    GRADIENT_ACCUMULATION_STEPS = 4
    MAX_GRAD_NORM = 1e7
    MODEL = "tf_efficientnet_b0"
    NUM_FROZEN_LAYERS = 39
    USE_CUSTOM_SPECS = True
    USE_PHASE_SPECS = True
    OVERLAPPING_SPECS = False
    NUM_WORKERS = mp.cpu_count()
    PRINT_FREQ = 20
    SEED = 545
    TRAIN_FULL_DATA = False
    K_FOLD_EXISTS = True
    TRAIN_SHUFFLE = True
    VISUALIZE = False
    WEIGHT_DECAY = 0.01

class config(config_train): pass

class paths:
    DATAPATH = os.path.expanduser("~/UM2024/eecs545/project/545-hms/")
    RAW_DATAPATH = DATAPATH
    OUTPUT_DIR = DATAPATH + 'train/'
    PRE_LOADED_EEGS = '/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy'
    PRE_LOADED_SPECTOGRAMS = '/kaggle/input/brain-spectrograms/specs.npy'
    TRAIN_CSV = RAW_DATAPATH + "train.csv"
    TRAIN_EEGS = "/kaggle/input/brain-eeg-spectrograms/EEG_Spectrograms/"
    TRAIN_SPECTOGRAMS = RAW_DATAPATH + "train_spectrograms/"
    

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Utils</b><a class='anchor' id='utils'></a> [↑](#top) 
# 
# ***
# 
# Utility functions.

# %%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s: float):
    "Convert to minutes."
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since: float, percent: float):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def get_logger(filename=paths.OUTPUT_DIR):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}train.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def plot_spectrogram(spectrogram_path: str):
    """
    Source: https://www.kaggle.com/code/mvvppp/hms-eda-and-domain-journey
    Visualize spectogram recordings from a parquet file.
    :param spectrogram_path: path to the spectogram parquet.
    """
    sample_spect = pd.read_parquet(spectrogram_path)
    
    split_spect = {
        "LL": sample_spect.filter(regex='^LL', axis=1),
        "RL": sample_spect.filter(regex='^RL', axis=1),
        "RP": sample_spect.filter(regex='^RP', axis=1),
        "LP": sample_spect.filter(regex='^LP', axis=1),
    }
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    label_interval = 5
    for i, split_name in enumerate(split_spect.keys()):
        ax = axes[i]
        img = ax.imshow(np.log(split_spect[split_name]).T, cmap='viridis', aspect='auto', origin='lower')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Log(Value)')
        ax.set_title(split_name)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        ax.set_yticks(np.arange(len(split_spect[split_name].columns)))
        ax.set_yticklabels([column_name[3:] for column_name in split_spect[split_name].columns])
        frequencies = [column_name[3:] for column_name in split_spect[split_name].columns]
        ax.set_yticks(np.arange(0, len(split_spect[split_name].columns), label_interval))
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()
    
    
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 

    
def sep():
    print("-"*100)
    

target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other':5}
num_to_label = {v: k for k, v in label_to_num.items()}
LOGGER = get_logger()
writer = None
seed_everything(config.SEED)

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Load Data</b><a class='anchor' id='load_data'></a> [↑](#top) 
# 
# ***
# 
# Load the competition's data.

# %%
df = pd.read_csv(paths.TRAIN_CSV)
df.insert(0, 'index', range(len(df)))
label_cols = df.columns[-6:]
print(f"Train cataframe shape is: {df.shape}")
print(f"Labels: {list(label_cols)}")
if config.VISUALIZE:
    df.head()

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Data pre-processing</b><a class='anchor' id='pre_processing'></a> [↑](#top) 
# 
# ***
# 
# ### <b><span style='color:#F1A424'>Create Non-Overlapping Eeg Id Train Data</span></b>
# 
# The competition data description says that test data does not have multiple crops from the same `eeg_id`. Therefore we will train and validate using only 1 crop per `eeg_id`. There is a discussion about this [here][1].
# 
# [1]: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/467021

# %%
if config.OVERLAPPING_SPECS:
    train_df = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({
        'spectrogram_id':'first',
        'spectrogram_label_offset_seconds':'min'
    })
    train_df.columns = ['spectogram_id','min']

    aux = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg({
        'spectrogram_label_offset_seconds':'max'
    })
    train_df['max'] = aux

    aux = df.groupby('eeg_id')[['patient_id']].agg('first')
    train_df['patient_id'] = aux

    aux = df.groupby('eeg_id')[label_cols].agg('sum')
    for label in label_cols:
        train_df[label] = aux[label].values
        
    y_data = train_df[label_cols].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train_df[label_cols] = y_data

    aux = df.groupby('eeg_id')[['expert_consensus']].agg('first')
    train_df['target'] = aux

    train_df = train_df.reset_index()
    print('Train non-overlapp eeg_id shape:', train_df.shape )
    if config.VISUALIZE:
        train_df.head()

else:
    train_df = df
    train_df.rename(columns={'expert_consensus':'target'}, inplace=True)
    y_data = train_df[label_cols].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train_df[label_cols] = y_data
    if config.VISUALIZE:
        train_df

# %% [markdown]
# ### <b><span style='color:#F1A424'>Read Train Spectrograms</span></b>
# 
# 
# First we need to read in all 11k train spectrogram files. Reading thousands of files takes 11 minutes with Pandas. Instead, we can read 1 file from my [Kaggle dataset here][1] which contains all the 11k spectrograms in less than 1 minute! To use my Kaggle dataset, set variable `READ_SPEC_FILES = False`. Thank you for upvoting my helpful [dataset][1] :-)
# 
# The resulting `all_spectograms` dictionary contains `spectogram_id` as keys (`int` keys) and the values are the spectogram sequences (as 2-dimensional `np.array`) of shape `(timesteps, 400)`.
# 
# Each spectogram is a parquet file. This parquet, when converted to a pandas dataframe, results in a dataframe of shape `(time_steps, 401)`. First column is the `time` column and the remaining 400 columns are the recordings. There are 400 columns because there are, respectively, 100 rows associated to the 4 recording regions of the EEG electrodes: `LL`, `RL`, `LP`, `RP`. Column names also include the frequency in heartz.
# 
# [1]: https://www.kaggle.com/datasets/cdeotte/brain-spectrograms

# %%
READ_SPEC_FILES = True

paths_spectograms = glob(paths.TRAIN_SPECTOGRAMS + "*.parquet")
print(f'There are {len(paths_spectograms)} spectrogram parquets')

def load_spectrogram(file_path):
    """
    Function to load a single spectrogram from a parquet file.
    """
    aux = pd.read_parquet(file_path)
    name = int(file_path.split("/")[-1].split('.')[0])
    spectrogram = aux.iloc[:,1:].values
    del aux
    return name, spectrogram

if READ_SPEC_FILES:
    all_spectrograms = {}
    with ProcessPoolExecutor() as executor:
        # Using map to ensure the order of results matches the order of submission
        results = executor.map(load_spectrogram, paths_spectograms)
        
        # Populate the all_spectrograms dictionary with results in order
        for name, spectrogram in results:
            all_spectrograms[name] = spectrogram
else:
    all_spectrograms = np.load(paths.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()
    
if config.VISUALIZE:
    idx = np.random.randint(0,len(paths_spectograms))
    spectrogram_path = paths_spectograms[idx]
    plot_spectrogram(spectrogram_path)

# %% [markdown]
# ### <b><span style='color:#F1A424'>Read EEG Spectrograms</span></b>
# 
# The resulting `all_eegs` dictionary contains `eeg_id` as keys (`int` keys) and the values are the eeg sequences (as 3-dimensional `np.array`) of shape `(128, 256, 4)`.
# 
# 

# %%
# %%time
# READ_EEG_SPEC_FILES = False

# paths_eegs = glob(paths.TRAIN_EEGS + "*.npy")
# print(f'There are {len(paths_eegs)} EEG spectograms')

# if READ_EEG_SPEC_FILES:
#     all_eegs = {}
#     for file_path in tqdm(paths_eegs):
#         eeg_id = file_path.split("/")[-1].split(".")[0]
#         eeg_spectogram = np.load(file_path)
#         all_eegs[eeg_id] = eeg_spectogram
# else:
#     all_eegs = np.load(paths.PRE_LOADED_EEGS, allow_pickle=True).item()

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Validation</b><a class='anchor' id='validation'></a> [↑](#top) 
# 
# ***
# 
# We train using `GroupKFold` on `patient_id`.

# %%
from sklearn.model_selection import KFold, GroupKFold

if config.K_FOLD_EXISTS:
    train_df = pd.read_csv('k_fold.csv')
else:
    gkf = GroupKFold(n_splits=config.FOLDS)
    for fold, (train_index, valid_index) in enumerate(gkf.split(train_df, train_df.target, train_df.patient_id)):
        train_df.loc[valid_index, "fold"] = int(fold)
if config.VISUALIZE:
    display(train_df.groupby('fold').size()), sep()
    display(train_df.head())

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Dataset</b><a class='anchor' id='dataset'></a> [↑](#top) 
# 
# ***
# 
# Create a custom `Dataset` to load data.
# 
# Our dataloader outputs both Kaggle spectrograms and EEG spectrogams as 8 channel image of size `(128, 256, 8)`
# 
# [1]: https://www.kaggle.com/code/cdeotte/efficientnetb0-starter-lb-0-43/comments#2617811

# %%
class CustomDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, config,
        augment: bool = False, mode: str = 'train',
        specs: Dict[int, np.ndarray] = all_spectrograms,
        eeg_specs: Dict[int, np.ndarray] = None #all_eegs
    ): 
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE_TRAIN
        self.augment = augment
        self.mode = mode
        self.kaggle_specs = all_spectrograms
        self.eeg_spectograms = eeg_specs
        
    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return len(self.df)
        
    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X) 
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


    def __data_generation(self, index):
        """
        Generates data containing batch_size samples.
        """
        X = np.zeros((128, 256, 12), dtype='float32')
        y = np.zeros(6, dtype='float32')
        img = np.ones((128,256), dtype='float32')
        row = self.df.iloc[index]
        major = row['index'] // 100
        minor = row['index'] % 100
        # if self.mode=='test': 
        #     r = 0
        # else: 
        #     r = int((row['min'] + row['max']) // 4)
        start = int((row['spectrogram_label_offset_seconds']) // 2)
        end = start + 300
            
        for region in range(4):
            img = self.kaggle_specs[row['spectrogram_id']][start:end, region*100:(region+1)*100].T #(f, t)
            
            # Log transform spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img-mu)/(std+ep)
            img = np.nan_to_num(img, nan=0.0)
            X[:, :, region] = cv2.resize(img, (256,128))

            if self.mode != 'test':
                y = row[label_cols].values.astype(np.float32)
        
        cust_spec = np.load(paths.DATAPATH + 'custom_specs/spec_{:04}.npy'.format(major+1), 'r')[minor]
        cust_spec = np.moveaxis(cust_spec, 0, -1)
        X[:, :, 4:8] = cust_spec

        phase = np.load(paths.DATAPATH + 'phase_spectrum/phase_{:04}.npy'.format(major+1), 'r')[minor].T
        phase = cv2.resize(phase, (256,128))
        X[:, :, 8:] = phase


        return X, y
    
    def __transform(self, img):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        img = transform(img)
        return img

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> DataLoader</b><a class='anchor' id='dataloader'></a> [↑](#top) 
# 
# ***
# 
# Below we display example dataloader spectrogram images.

# %%
train_dataset = CustomDataset(train_df, config, mode="train")
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE_TRAIN,
    shuffle=config.TRAIN_SHUFFLE,
    num_workers=config.NUM_WORKERS, drop_last=True
)
X, y = train_dataset[0]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# %% [markdown]
# ### <b><span style='color:#F1A424'> Visualize DataLoader</span></b>
# 

# %%
if config.VISUALIZE:
    ROWS = 2
    COLS = 3
    for (X, y) in train_loader:
        plt.figure(figsize=(20,8))
        for row in range(ROWS):
            for col in range(COLS):
                plt.subplot(ROWS, COLS, row*COLS + col+1)
                t = y[row*COLS + col]
                img = X[row*COLS + col, :, :, 0]
                mn = img.flatten().min()
                mx = img.flatten().max()
                img = (img-mn)/(mx-mn)
                plt.imshow(img)
                tars = f'[{t[0]:0.2f}'
                for s in t[1:]:
                    tars += f', {s:0.2f}'
                eeg = train_df.eeg_id.values[row*config.BATCH_SIZE_TRAIN + row*COLS + col]
                plt.title(f'EEG = {eeg}\nTarget = {tars}',size=12)
                plt.yticks([])
                plt.ylabel('Frequencies (Hz)',size=14)
                plt.xlabel('Time (sec)',size=16)
        plt.show()
        break

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Model</b><a class='anchor' id='model'></a> [↑](#top) 
# 
# ***
# 
# We will be using the [timm](https://github.com/huggingface/pytorch-image-models) library for our models.
# 
# Our models receives both Kaggle spectrograms and EEG spectrograms from our data loader. We then reshape these 8 spectrograms into 1 large flat image and feed it into EfficientNet.

# %%
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6, pretrained: bool = True):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.USE_PHASES = True
        self.model = timm.create_model(
            config.MODEL,
            pretrained=pretrained,
            drop_rate = 0.1,
            drop_path_rate = 0.2,
        )
        if config.FREEZE:
            for i,(name, param) in enumerate(list(self.model.named_parameters())\
                                             [0:config.NUM_FROZEN_LAYERS]):
                param.requires_grad = False

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes)
        )

    def __reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """ 
        # === Get spectograms ===
        spectograms = [x[:, :, :, i:i+1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=1)
        
        # === Get EEG spectograms ===
        eegs = [x[:, :, :, i:i+1] for i in range(4,8)]
        eegs = torch.cat(eegs, dim=1)

        # === Get phase spectograms ===
        phases = [x[:, :, :, i:i+1] for i in range(8,12)]
        phases = torch.cat(phases, dim=1)
        
        # === Reshape (512,512,3) ===
        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS & self.USE_PHASES:
            x = torch.cat([spectograms, eegs, phases], dim=2)
        elif self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectograms
            
        x = torch.cat([x,x,x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x
    
    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Scheduler</b><a class='anchor' id='scheduler'></a> [↑](#top) 
# 
# ***
# 
# We will train our model with a Step Train Schedule for 4 epochs. First 2 epochs are LR=1e-3. Then epochs 3 and 4 use LR=1e-4 and 1e-5 respectively. (Below we also provide a Cosine Train Schedule if you want to experiment with it. Note it is not used in this notebook).

# %%
from torch.optim.lr_scheduler import OneCycleLR

EPOCHS = config.EPOCHS
BATCHES = len(train_loader)
steps = []
lrs = []
optim_lrs = []
model = CustomModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=config.EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.05,
    anneal_strategy="cos",
    final_div_factor=100,
)
for epoch in range(EPOCHS):
    for batch in range(BATCHES):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
        steps.append(epoch * BATCHES + batch)

max_lr = max(lrs)
min_lr = min(lrs)
print(f"Maximum LR: {max_lr} | Minimum LR: {min_lr}")
if config.VISUALIZE:
    plt.figure()
    plt.plot(steps, lrs, label='OneCycle')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.show()


# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Loss Function</b><a class='anchor' id='loss'></a> [↑](#top) 
# 
# ***
# 
# In PyTorch's [KLDivLoss][1], the reduction parameter determines how the loss is aggregated across different dimensions. Two common options are `mean` and `batchmean`.
# 
# - `reduction`='mean': When reduction is set to "mean", the Kullback-Leibler Divergence loss is computed and then averaged over all the elements in the input tensor. The result is a scalar value representing the mean loss.
# - `reduction`='batchmean': When reduction is set to "batchmean", the Kullback-Leibler Divergence loss is computed independently for each item in the batch, and then the mean is taken over the batch dimension. This is useful when you have a batch of samples, and you want the average loss per sample.
# 
# [1]: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
# 

# %%
# # === Reduction = "mean" ===
# criterion = nn.KLDivLoss(reduction="mean")
# y_pred = F.log_softmax(torch.randn(6, 2, requires_grad=True), dim=1)
# y_true = F.softmax(torch.rand(6, 2), dim=1)
# print(f"Predictions: {y_pred}")
# print(f"Targets: {y_true}")
# output = criterion(y_pred, y_true)
# print(f"Output: {output}")

# print("\n", "="*100, "\n")

# # === Reduction = "batchmean" ===
# criterion = nn.KLDivLoss(reduction="batchmean")
# print(f"Predictions: {y_pred}")
# print(f"Targets: {y_true}")
# output = criterion(y_pred, y_true)
# print(f"Output: {output}")

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Train and Validation Functions</b><a class='anchor' id='functions'></a> [↑](#top) 
# 
# ***
# 
# We train using Group KFold on patient id. If `LOAD_MODELS_FROM = None`, then we will train new models in this notebook version. Otherwise we will load saved models from the path `LOAD_MODELS_FROM`.

# %%
def train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """One epoch training pass."""
    global writer
    model.train() 
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    
    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, (X, y) in enumerate(tqdm_train_loader):
            gc.collect()
            torch.cuda.empty_cache()
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.cuda.amp.autocast(enabled=config.AMP):
                y_preds = model(X) 
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                scheduler.step()
            end = time.time()

            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(epoch+1, step, len(train_loader), 
                              remain=timeSince(start, float(step+1)/len(train_loader)),
                              loss=losses,
                              grad_norm=grad_norm,
                              lr=scheduler.get_last_lr()[0]))
                writer.add_scalar('Training loss', loss.detach().item(), epoch * len(train_loader) + step)

    writer.flush()
    return losses.avg


def valid_epoch(valid_loader, model, criterion, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    losses = AverageMeter()
    prediction_dict = {}
    preds = []
    start = end = time.time()
    with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
        for step, (X, y) in enumerate(tqdm_valid_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.no_grad():
                y_preds = model(X)
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy())
            end = time.time()

            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(valid_loader)-1):
                print('EVAL: [{0}/{1}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      .format(step, len(valid_loader),
                              remain=timeSince(start, float(step+1)/len(valid_loader)),
                              loss=losses))
                
    prediction_dict["predictions"] = np.concatenate(preds)
    return losses.avg, prediction_dict

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Train Loop</b><a class='anchor' id='train_loop'></a> [↑](#top) 
# 
# ***

# %%
def train_loop(df, fold):
    
    LOGGER.info(f"========== Fold: {fold} training ==========")
    global writer
    writer = SummaryWriter(f"{paths.OUTPUT_DIR}tb_fold{fold}/")

    # ======== SPLIT ==========
    train_folds = df[df['fold'] != fold].reset_index(drop=True)
    valid_folds = df[df['fold'] == fold].reset_index(drop=True)
    
    # ======== DATASETS ==========
    train_dataset = CustomDataset(train_folds, config, mode="train", augment=False)
    valid_dataset = CustomDataset(valid_folds, config, mode="train", augment=False)
    
    # ======== DATALOADERS ==========
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=config.TRAIN_SHUFFLE,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_VALID,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)
    
    # ======== MODEL ==========
    model = CustomModel(config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=config.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )

    # ======= LOSS ==========
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    best_loss = np.inf
    # ====== ITERATE EPOCHS ========
    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # ======= TRAIN ==========
        avg_train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # ======= EVALUATION ==========
        avg_val_loss, prediction_dict = valid_epoch(valid_loader, model, criterion, device)
        predictions = prediction_dict["predictions"]
        
        # ======= SCORING ==========
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        writer.add_scalars(f'Fold {fold} - train & eval loss', {'Training loss': avg_train_loss, 'Eval loss': avg_val_loss}, epoch+1)
        writer.flush()
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        paths.OUTPUT_DIR + f"/{config.MODEL.replace('/', '_')}_fold_{fold}_best.pth")

    predictions = torch.load(paths.OUTPUT_DIR + f"/{config.MODEL.replace('/', '_')}_fold_{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[target_preds] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    writer.flush()
    writer.close()
    
    return valid_folds

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Train Full Data</b><a class='anchor' id='train_full'></a> [↑](#top) 
# 
# ***

# %%
def train_loop_full_data(df):
    train_dataset = CustomDataset(df, config, mode="train", augment=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    model = CustomModel(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=config.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )
    criterion = nn.KLDivLoss(reduction="batchmean")
    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        avg_train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  time: {elapsed:.0f}s')
        torch.save(
            {'model': model.state_dict()},
            paths.OUTPUT_DIR + f"/{config.MODEL.replace('/', '_')}_epoch_{epoch}.pth")
    torch.cuda.empty_cache()
    gc.collect()
    return _

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Train</b><a class='anchor' id='train'></a> [↑](#top) 
# 
# ***

# %%
def get_result(oof_df):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(oof_df[label_cols].values)
    preds = torch.tensor(oof_df[target_preds].values)
    preds = torch.log(preds)
    result = kl_loss(preds, labels)
    return result

if not config.TRAIN_FULL_DATA:
    oof_df = pd.DataFrame()
    for fold in range(config.FOLDS):
        if fold in [0, 1, 2, 3, 4]:
            _oof_df = train_loop(train_df, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== Fold {fold} result: {get_result(_oof_df)} ==========")
            print(f"========== Fold {fold} result: {get_result(_oof_df)} ==========")
    oof_df = oof_df.reset_index(drop=True)
    LOGGER.info(f"========== CV: {get_result(oof_df)} ==========")
    oof_df.to_csv(paths.OUTPUT_DIR + '/oof_df.csv', index=False)
else:
    train_loop_full_data(train_df)


