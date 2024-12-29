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


class String2Index:
    def __init__(self):
        self.mapping = {}
        self.reverse_mapping = {}
    
    def fit(self, data):
        unique_data = np.unique(data)
        for i, d in enumerate(unique_data):
            self.mapping[d] = i
            self.reverse_mapping[i] = d
    
    def transform(self, data):
        return np.array([self.mapping[d] for d in data])
    
    def reverse_transform(self, data):
        return np.array([self.reverse_mapping[d] for d in data])


class AbideROIDataset(Dataset):
    def __init__(self, csv, data_root, n_views, atlas='cc400', task='DX', transforms=None, cp="", cnp=""):
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

        # self.num_fea_names = ['AGE_AT_SCAN', 'HANDEDNESS_SCORES', 'BMI']
        # self.str_fea_names = ['SITE_ID', 'SEX', 'HANDEDNESS_CATEGORY', 'CURRENT_MED_STATUS']
        self.category_phenotype_names = cp.split(', ')
        self.continuous_phenotype_names = cnp.split(', ')
        
        self.cp_fea = csv[self.category_phenotype_names].values
        self.cnp_fea = csv[self.continuous_phenotype_names].values

        # TODO: deal with the missing values
        self.cp_fea[self.cp_fea == -9999] = 'unk'
        self.cp_fea[self.cp_fea == '-9999'] = 'unk'
        # special case. The real-world data!
        self.cp_fea[self.cp_fea == '`'] = 'unk'
        self.cp_fea = self._string2index(self.cp_fea)

        self.cnp_fea[self.cnp_fea == -9999] = 0
        self.cnp_fea[self.cnp_fea == '-9999'] = 0
        # normalize the numerical features by each column
        self.cnp_fea = (self.cnp_fea - self.cnp_fea.mean(axis=0)) / self.cnp_fea.std(axis=0)

        self.num_cp = len(self.category_phenotype_names) + 1 # add label information
        self.num_cnp = len(self.continuous_phenotype_names)


    @staticmethod
    def _string2index(data):
        transformed_data = np.empty(data.shape, dtype=int)
        for col in range(data.shape[1]):
            unique_values = {value: idx for idx, value in enumerate(set(data[:, col]))}
            # Map 'unk' to -1
            unique_values['unk'] = -1
        
            for row in range(data.shape[0]):
                transformed_data[row, col] = unique_values.get(data[row, col], -1)
    
        return transformed_data

    
    def __len__(self):
        return len(self.labels)
        
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        file_id = self.filenames[idx]
        cnp_fea = self.cnp_fea[idx]
        cp_fea = self.cp_fea[idx]
        file_path = os.path.join(self.data_root, self.filenames[idx] + self.suffix)
        roi = pd.read_csv(file_path, sep='\t').values.T # [T, N] -> [N, T]
        if self.transforms:
            roi = pad_sequence([torch.from_numpy(self.transforms(roi).T) for _ in range(self.n_views)], batch_first=True) # [V, T, N]
            roi = rearrange(roi, 'v t n -> t v n') # [V, N, T] -> [T, V, N]

            # make the labels and features consistent with the views
            label = np.stack([label for _ in range(self.n_views)], axis=0) # [V]
            

        else:
            # [N, T] -> [T, 1, N]
            roi = torch.from_numpy(roi.T).unsqueeze(1).float()

        return roi, label, cnp_fea, cp_fea

    @staticmethod
    def collate_fn(batch):
        data, label, cnp_fea, cp_fea = list(zip(*batch))
        # pad the sequence on T
        data = pad_sequence(data, batch_first=True).float()
        label = torch.from_numpy(np.array(label)).long()
        cnp_fea = torch.from_numpy(np.array(cnp_fea)).float()
        cp_fea = torch.from_numpy(np.array(cp_fea)).float()
        return data, label, cnp_fea, cp_fea
