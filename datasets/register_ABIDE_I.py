import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchio as tio
from nilearn import image


def register_ABIDE_I(data_root, output_dir):
    csv_path = os.path.join(data_root, 'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError('Phenotypic file not found. Please download ABIDE I dataset and place it in the correct directory.')
    official_csv = pd.read_csv(csv_path)[['SUB_ID', 'SITE_ID', 'FILE_ID', 'DX_GROUP', 'DSM_IV_TR']]
    # According to ABIDDE, the DX_GROUP is 1 for autism and 2 for control
    # rearrange the DX to 0 for control and 1 for autism to be consistent with other datasets
    official_csv['DX_GROUP'] = 2 - official_csv['DX_GROUP']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    roi_df = pd.DataFrame(columns=['SUB_ID', 'SITE_ID', 'FILE_ID', 'DX_GROUP', 'DSM_IV_TR', 'num_roi', 'seq_len'])

    # Default preprocessing 
    data_dir = os.path.join(data_root, 'ABIDE_pcp', 'cpac', 'nofilt_noglobal')
    suffix = '_rois_cc400.1D'
    for index, row in tqdm(official_csv.iterrows()):
        file_path = os.path.join(data_dir, row['FILE_ID'] + suffix)
        if os.path.exists(file_path):
            roi = pd.read_csv(file_path, sep='\t').values
            roi_df = roi_df._append({'SUB_ID': row['SUB_ID'], 'SITE_ID': row['SITE_ID'], 'FILE_ID': row['FILE_ID'], 'DX_GROUP': row['DX_GROUP'], 'DSM_IV_TR': row['DSM_IV_TR'], 'num_roi': roi.shape[1], 'seq_len': roi.shape[0]}, ignore_index=True)

        roi_df.to_csv(os.path.join(output_dir, 'ABIDEI_ROI.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Register ABIDE I dataset')
    parser.add_argument('--src', type=str, default='/data', help='Root directory of the dataset')
    parser.add_argument('--dst', type=str, default='./splits', help='Output directory')
    args = parser.parse_args()
    register_ABIDE_I(args.src, args.dst)
