# 545-hms

# code setup user TODOs
- pip install mne

# Files
-preprocessing_EEG.ipynb
  - place in hms-harmful-brain-activity-classification directory 

# Data TODOs
- EEG
   - [x] Preprocess 50s for 106800 EEG samples in Kaggle Dataset using MNE (To check and make sure it can run for all)
   - [ ] Preprocess External Dataset
   - [ ] Additional Preprocessing (e.g. imputation/outlier detection)

- Spectrogram 
   - [ ] Create Spectrogram from New EEG
   - [ ] Preprocess Kaggle Spectrogram
        - [x] Get 10 min for 106800 EEG samples in Kaggle Dataset      
- [ ] Cross-validation

# Training Model TODOs
1. [ ] Take pretrained weights from someone’s EEG model
2. [ ] Take pretrained weights from someone’s spectrogram model
3. [ ] Play around with fusion methods and classification to validate this part
4. [ ] Go back and train our own models/determine our own encoder architectures


