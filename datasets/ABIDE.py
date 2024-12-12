import os
import torch
import pandas as pd
import numpy as np
import torchio as tio
from nilearn import image
from einops import rearrange
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder


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
    def __init__(self, csv, data_root, n_views, atlas='cc400', task='DX', transforms=None):
        self.csv = csv
        # keep consistent with the nan filling strategy
        csv = csv.fillna(-9999)

        self.filenames = csv['FILE_ID'].values
        self.labels = csv['DX_GROUP'].values
        self.suffix = f"_rois_{atlas}.1D"
        self.n_views = n_views
        self.data_root = data_root
        self.transforms = transforms
        self.n_classes = len(np.unique(self.labels))

        self.num_fea_names = ['AGE_AT_SCAN', 'HANDEDNESS_SCORES', 'BMI']
        self.str_fea_names = ['SITE_ID', 'SEX', 'HANDEDNESS_CATEGORY', 'CURRENT_MED_STATUS']

        self.num_fea = csv[self.num_fea_names].values
        self.str_fea = csv[self.str_fea_names].values

        # TODO: deal with the missing values
        self.num_fea[self.num_fea == -9999] = 0
        self.num_fea[self.num_fea == '-9999'] = 0
        self.str_fea[self.str_fea == -9999] = 'unk'
        self.str_fea[self.str_fea == '-9999'] = 'unk'
        # special case. The real-world data!
        self.str_fea[self.str_fea == '`'] = 'unk'

        self.onehot = OneHotEncoder()
        self.str_fea = self.onehot.fit_transform(self.str_fea).toarray()

        self.onehot_expand = [len(self.onehot.categories_[i]) for i in range(len(self.str_fea_names))]
    
    def __len__(self):
        return len(self.labels)
        
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        file_id = self.filenames[idx]
        num_fea = self.num_fea[idx]
        str_fea = self.str_fea[idx]
        file_path = os.path.join(self.data_root, self.filenames[idx] + self.suffix)
        roi = pd.read_csv(file_path, sep='\t').values.T # [T, N] -> [N, T]
        if self.transforms:
            roi = pad_sequence([torch.from_numpy(self.transforms(roi).T) for _ in range(self.n_views)], batch_first=True) # [V, T, N]
            roi = rearrange(roi, 'v t n -> t v n') # [V, N, T] -> [T, V, N]

            # make the labels and features consistent with the views
            label = np.stack([label for _ in range(self.n_views)], axis=0) # [V]
            num_fea = np.stack([num_fea for _ in range(self.n_views)], axis=0) # [V, n_num]
            str_fea = np.stack([str_fea for _ in range(self.n_views)], axis=0) # [V, n_str]
            

        else:
            # [N, T] -> [T, 1, N]
            roi = torch.from_numpy(roi.T).unsqueeze(1).float()

        return roi, label, num_fea, str_fea

    @staticmethod
    def collate_fn(batch):
        data, label, num_fea, str_fea = list(zip(*batch))
        # pad the sequence on T
        data = pad_sequence(data, batch_first=True).float()
        label = torch.from_numpy(np.array(label)).long()
        num_fea = torch.from_numpy(np.array(num_fea)).float()
        str_fea = torch.from_numpy(np.array(str_fea)).float()
        return data, label, num_fea, str_fea
