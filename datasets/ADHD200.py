import os
import torch
import pandas as pd
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder


class AdhdROIDataset(Dataset):
    def __init__(self, csv, data_root, atlas, n_views, transforms, filter='Yes', cp="", cnp="", task='DX'):
        super().__init__()
        self.filter = filter
        if filter == 'both':
            self.csv = csv
        elif filter == 'Yes':
            self.csv = csv[csv['Filename'].str.startswith('sf')]
        elif filter == 'No':
            self.csv = csv[csv['Filename'].str.startswith('sn')]
        else:
            raise ValueError(f'Invalid filter: {filter}')
        self.csv = self.csv.fillna(-999)
        self.csv['Subject_ID'] = self.csv['Subject_ID'].apply(lambda x: f'{x:07d}')
        self.data_root = data_root
        self.atlas = atlas
        self.data_path = os.path.join(data_root, atlas)
        self.n_views = n_views
        self.transforms = transforms
        self.task = task
        self.labels = self.csv['DX'].values
        if self.task == 'DX': # binary classification
            self.labels[self.labels > 0] = 1 
        self.n_classes = len(np.unique(self.labels))


        self.cp_columns = cp.split(', ')
        self.cnp_columns = cnp.split(', ')

        self.cp_labels = self.csv[self.cp_columns].values.astype(int)
        self.cnp_labels = self.csv[self.cnp_columns].values.astype(float)

        self.cp_labels[self.cp_labels < 0] = -1
        self.cnp_labels[self.cnp_labels < 0] = -1

        self.num_cp = len(self.cp_columns) + 1
        self.num_cnp = len(self.cnp_columns)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        subject = row['Subject_ID']
        filename = row['Filename']
        label = self.labels[idx]
        cp_label = self.cp_labels[idx]
        cnp_label = self.cnp_labels[idx]
        file_path = os.path.join(self.data_path, subject, filename)
        roi = pd.read_csv(file_path, sep='\t').values[:, 2:].astype(float).T # drop the first two columns
        if self.transforms:
            temp = [torch.from_numpy(self.transforms(roi).T) for _ in range(self.n_views - 1)]
            # append the original view
            temp.append(torch.from_numpy(roi.T))
            roi = pad_sequence(temp, batch_first=True) # [V, T, N]
            roi = rearrange(roi, 'v t n -> t v n') # [V, N, T] -> [T, V, N]
            
        else:
            # [N, T] -> [T, 1, N]
            roi = torch.from_numpy(roi.T).unsqueeze(1).float()
        return roi, label, cnp_label, cp_label

    @staticmethod
    def collate_fn(batch):
        data, label, cnp_label, cp_label = list(zip(*batch))
        # pad the sequence on T
        data = pad_sequence(data, batch_first=True).float()
        label = torch.from_numpy(np.array(label)).long()
        cnp_label = torch.from_numpy(np.array(cnp_label)).float()
        cp_label = torch.from_numpy(np.array(cp_label)).float()
        return {'x': data, 'label': label, 'cnp_label': cnp_label, 'cp_label': cp_label}
