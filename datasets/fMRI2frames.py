# slide the fMRI into frames and save them for faster frame-level training
import os
import copy
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchio as tio
from nilearn import image


def slice_fMRI(data_root, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = os.path.join(data_root, 'ABIDE_pcp', 'cpac', 'nofilt_noglobal')
    suffix = '_func_preproc.nii.gz'
    filenames = [f for f in os.listdir(data_dir) if f.endswith(suffix)]
    for file in tqdm(filenames, desc='fMRI', position=0):
        file_id = file.replace(suffix, '')
        dst_dir = os.path.join(output_dir, file_id)
        os.makedirs(dst_dir, exist_ok=True)

        file_path = os.path.join(data_dir, file)
        img = tio.ScalarImage(file_path)
        img_data = copy.deepcopy(img.data)
        for i in tqdm(range(img.shape[0]), desc='frames', position=1, leave=False):
            img.set_data(img_data[i][np.newaxis, ...])
            img.save(os.path.join(dst_dir, f'frame_{i}.nii.gz'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Register ABIDE I dataset')
    parser.add_argument('--data_root', type=str, default='/home/featurize/data', help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='/home/featurize/data/ABIDEI_frames', help='Output directory')
    args = parser.parse_args()
    slice_fMRI(args.data_root, args.output_dir)
