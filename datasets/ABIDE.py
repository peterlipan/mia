import os
import torch
import pandas as pd
import numpy as np
import torchio as tio
from nilearn import image
from torch.utils.data import Dataset


class AbideFrameDataset(Dataset):
    def __init__(self, csv, data_root, suffix='_func_preproc.nii.gz', task='DX', transforms=None):
        self.csv = csv
        self.filenames = csv['FILE_ID'].values
        self.labels = csv['DX_GROUP'].values
        self.frame_id = csv['FRAME_ID'].values
        self.suffix = suffix
        self.data_root = data_root
        self.transforms = transforms
        self.n_classes = len(np.unique(self.labels))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = os.path.join(self.data_root, self.filenames[idx] + self.suffix)
        # TODO: check if the transformation from nilearn to torchio is correct
        fmri = image.load_img(file_path)
        frame = image.index_img(fmri, self.frame_id[idx])
        frame = tio.ScalarImage(tensor=frame.get_fdata()[np.newaxis, ...], affine=frame.affine)
        if self.transforms:
            frame = self.transforms(frame)

        return frame, label
