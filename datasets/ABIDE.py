import os
import torch
import pandas as pd
import numpy as np
import torchio as tio
from nilearn import image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class AbideFrameDataset(Dataset):
    def __init__(self, csv, data_root, suffix='_func_preproc.nii.gz', task='DX', transforms=None, mode='frames'):
        self.csv = csv
        self.filenames = csv['FILE_ID'].values
        self.labels = csv['DX_GROUP'].values
        self.frame_id = csv['FRAME_ID'].values
        self.suffix = suffix
        self.data_root = data_root
        self.transforms = transforms
        self.n_classes = len(np.unique(self.labels))
        self.mode = mode
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        file_id = self.filenames[idx]
        frame_id = self.frame_id[idx]
        frame = None
        if self.mode == 'fMRI':
            file_path = os.path.join(self.data_root, self.filenames[idx] + self.suffix)
            # TODO: make the loading more efficient
            frame = tio.ScalarImage(file_path)
            frame.set_data(frame.data[frame_id][np.newaxis, ...])
        elif self.mode == 'frames':
            frame_path = os.path.join(self.data_root, file_id, f'frame_{frame_id}.nii.gz')
            frame = tio.ScalarImage(frame_path)
        if self.transforms:
            frame = self.transforms(frame)
        
        # frame: [1, H, W, D]
        return frame, label


class AbideFmriDataset(Dataset):
    def __init__(self, csv, data_root, suffix='_func_preproc.nii.gz', task='DX', transforms=None, mode='frames'):
        self.csv = csv
        self.filenames = csv['FILE_ID'].values
        self.labels = csv['DX_GROUP'].values
        self.suffix = suffix
        self.data_root = data_root
        self.transforms = transforms
        self.n_classes = len(np.unique(self.labels))
        self.mode = mode
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        file_id = self.filenames[idx]
        file_path = os.path.join(self.data_root, self.filenames[idx] + self.suffix)
        fmri = tio.ScalarImage(file_path)
        if self.transforms:
            fmri = self.transforms(fmri)
        # fmri: [T, H, W, D] -> [T, C, H, W, D] (C = 1)
        fmri = fmri.unsqueeze(1)

        return fmri, label

    @staticmethod
    def collate_fn(batch):
        data, label = list(zip(*batch))
        # pad the sequence on T
        data = pad_sequence(data, batch_first=True)
        label = torch.tensor(label)
        return data, label


class AbideFmriDataset(Dataset):
    def __init__(self, csv, data_root, suffix='_func_preproc.nii.gz', task='DX', transforms=None, mode='frames'):
        self.csv = csv
        self.filenames = csv['FILE_ID'].values
        self.labels = csv['DX_GROUP'].values
        self.suffix = suffix
        self.data_root = data_root
        self.transforms = transforms
        self.n_classes = len(np.unique(self.labels))
        self.mode = mode
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        file_id = self.filenames[idx]
        file_path = os.path.join(self.data_root, self.filenames[idx] + self.suffix)
        fmri = tio.ScalarImage(file_path)
        if self.transforms:
            fmri = self.transforms(fmri)
        # fmri: [T, H, W, D] -> [T, C, H, W, D] (C = 1)
        fmri = fmri.unsqueeze(1)

        return fmri, label

    @staticmethod
    def collate_fn(batch):
        data, label = list(zip(*batch))
        # pad the sequence on T
        data = pad_sequence(data, batch_first=True)
        label = torch.tensor(label)
        return data, label


class AbideROIDataset(Dataset):
    def __init__(self, csv, data_root, seq_len, suffix='_rois_cc400.1D', task='DX', transforms=None):
        self.csv = csv
        self.filenames = csv['FILE_ID'].values
        self.labels = csv['DX_GROUP'].values
        self.suffix = suffix
        self.data_root = data_root
        self.seq_len = seq_len
        self.transforms = transforms
        self.n_classes = len(np.unique(self.labels))
    
    def __len__(self):
        return len(self.labels)

    def trunc_or_pand(self, data):
        if data.shape[-1] > self.seq_len:
            return data[..., :self.seq_len]
        else:
            return np.pad(data, ((0, 0), (0, self.seq_len - data.shape[-1])), 'constant', constant_values=0)
        
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        file_id = self.filenames[idx]
        file_path = os.path.join(self.data_root, self.filenames[idx] + self.suffix)
        roi = pd.read_csv(file_path, sep='\t').values.T # [seq_len, num_roi] -> [num_roi, seq_len]
        roi = self.trunc_or_pand(roi)
        roi = torch.from_numpy(roi).float()
        label = torch.tensor(label).long()

        return roi, label
