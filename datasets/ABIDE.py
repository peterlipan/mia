import torch
import pandas as pd
import numpy as np
from nilearn import image
from torch.utils.data import Dataset


class AbideFrameDataset(Dataset):
    def __init__(self, csv, data_root, suffix='_func_preproc.nii.gz'):
        self.csv = csv
        self.filenames = csv['FILE_ID'].values
        self.labels = csv['DX_GROUP'].values
        self.frame_id = csv['FRAME_ID'].values
        self.suffix = suffix
        self.data_root = data_root
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_root, self.filenames[idx] + self.suffix)
        fmri = image.load_img(file_path)
        frame = image.index_img(fmri, self.frame_id[idx])
        return torch.from_numpy(file.get_fdata()).float(), torch.tensor(self.labels[idx]).long()