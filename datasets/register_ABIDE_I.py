import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchio as tio
from nilearn import image
from sklearn.model_selection import StratifiedShuffleSplit


def splite_dataset(csv, output_dir):
    # Based on the BrainNetworkTransformer
    # https://github.com/Wayfear/BrainNetworkTransformer/blob/8a588aadad0166209269fa114e5df4e42209e207/source/dataset/dataloader.py#L49
    # Split the dataset into train, val, and test as 70%, 10%, and 20% respectively
    
    sub_id = csv['SUB_ID'].values
    labels = csv['DX_GROUP'].values
    stratified = csv['SITE_ID'].values
    n_subs = len(sub_id)

    train_size = 706
    val_size = 101
    test_size = 202

    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

    split = StratifiedShuffleSplit(n_splits=1, test_size=val_size+test_size, train_size=train_size, random_state=42)
    for train_idx, test_val_idx in split.split(sub_id, stratified):
        train_sub_id = sub_id[train_idx]
        test_val_sub_id = sub_id[test_val_idx]
        stratified = stratified[test_val_idx]
        
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for test_idx, val_idx in split2.split(test_val_sub_id, stratified):
        test_sub_id = test_val_sub_id[test_idx]
        val_sub_id = test_val_sub_id[val_idx]
    
    train_df = csv[csv['SUB_ID'].isin(train_sub_id)]
    val_df = csv[csv['SUB_ID'].isin(val_sub_id)]
    test_df = csv[csv['SUB_ID'].isin(test_sub_id)]

    train_df.to_csv(os.path.join(output_dir, 'ABIDE_Training.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'ABIDE_Validation.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'ABIDE_Testing.csv'), index=False)


def register_ABIDE_I(data_root, output_dir):
    csv_path = os.path.join(data_root, 'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError('Phenotypic file not found. Please download ABIDE I dataset and place it in the correct directory.')
    official_csv = pd.read_csv(csv_path)
    # According to ABIDDE, the DX_GROUP is 1 for autism and 2 for control
    # rearrange the DX to 0 for control and 1 for autism to be consistent with other datasets
    official_csv['DX_GROUP'] = 2 - official_csv['DX_GROUP']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    roi_df = pd.DataFrame(columns=['SUB_ID', 'SITE_ID', 'FILE_ID', 'DX_GROUP', 'DSM_IV_TR', 'num_roi', 'seq_len'])

    # Default preprocessing 
    data_dir = os.path.join(data_root, 'ABIDE_pcp', 'cpac', 'filt_noglobal')
    suffix = f'_rois_{args.atlas}.1D'
    for index, row in tqdm(official_csv.iterrows()):
        file_path = os.path.join(data_dir, row['FILE_ID'] + suffix)
        if os.path.exists(file_path):
            roi = pd.read_csv(file_path, sep='\t').values
            # NOTE: BrainNetworkTransformer has dropped those files (n=26) of which time length < 100
            if roi.shape[0] < 100:
                continue
            roi_df = roi_df._append({'SUB_ID': row['SUB_ID'], 'SITE_ID': row['SITE_ID'], 'FILE_ID': row['FILE_ID'], 'DX_GROUP': row['DX_GROUP'], 'DSM_IV_TR': row['DSM_IV_TR'], 'num_roi': roi.shape[1], 'seq_len': roi.shape[0]}, ignore_index=True)
    
    # join the phenotypic data with the roi data
    off_cols = official_csv.columns.tolist()
    start_idx = off_cols.index('AGE_AT_SCAN')
    end_idx = off_cols.index('BMI')
    selected_cols = ['SUB_ID'] + off_cols[start_idx:end_idx+1]
    part_df = official_csv[selected_cols]
    final_df = pd.merge(roi_df, part_df, on='SUB_ID', how='left')
    final_df.to_csv(os.path.join(output_dir, f'ABIDEI_roi_{args.atlas}_info.csv'), index=False)
    return final_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Register ABIDE I dataset')
    parser.add_argument('--src', type=str, default='/data', help='Root directory of the dataset')
    parser.add_argument('--dst', type=str, default='./splits', help='Output directory')
    parser.add_argument('--atlas', type=str, default='cc400')
    args = parser.parse_args()
    csv = register_ABIDE_I(args.src, args.dst)
    splite_dataset(csv, args.dst)

