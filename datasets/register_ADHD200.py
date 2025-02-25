"""
This script is to register and organize the ADHD 200 dataset preprocessed by Athena.
For a specific atlas (for example, AAL), please download:
ADHD200_AAL_TCs_filtfix.tar.gz (ROI time series of training samples)
ADHD200_AAL_TCs_TestRelease.tar (ROI time series of test samples)
and
allSubs_testSet_phenotypic_dx.csv, containing the label of test samples which are not included in the Athena's preprocessed data.
"""
import os
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm

atlas2num_roi = {'cc400': 351}


class Site:
    def __init__(self, name, path, test=False, target_num_roi=351):
        self.name = name
        self.path = path
        self.target_num_roi = target_num_roi
        self.pheno_path = os.path.join(path, f'{name}_phenotypic.csv') if not test else os.path.join(path, f'{name}_TestRelease_phenotypic.csv')
        assert os.path.exists(self.pheno_path), f'File not found: {self.pheno_path}'
        self.file_df = self.gather_file_df()
        
    
    @property
    def df(self):
        _df = pd.read_csv(self.pheno_path)
        self.sub_col = 'ScanDir ID' if 'ScanDir ID' in _df else 'ScanDirID' # for WashU
        # transform the subjects to str and keep in %07d
        _df[self.sub_col] = _df[self.sub_col].apply(lambda x: f'{x:07d}')
        return _df

    @property
    def subjects(self):
        return self.df[self.sub_col].values

    def gather_file_df(self):
        df = pd.DataFrame(columns=['Filename', 'Num_ROI', 'Seq_len'] + list(self.df.columns))
        for subject in self.subjects:
            sub_attr = self.df[self.df[self.sub_col] == subject].iloc[0].to_dict()
            sub_path = os.path.join(self.path, str(subject))
            for item in os.listdir(sub_path):
                if item.endswith('.1D'):
                    sub_attr['Filename'] = item
                    roi = pd.read_csv(os.path.join(sub_path, item), sep='\t').values[:, 2:]
                    num_roi = roi.shape[1]
                    seq_len = roi.shape[0]
                    sub_attr['Num_ROI'] = num_roi
                    sub_attr['Seq_len'] = seq_len
                    if num_roi < self.target_num_roi:
                        print(f'Warning: {subject} has {num_roi} ROIs instead of {self.target_num_roi}')
                        continue
                    if seq_len < 50:
                        print(f'Warning: {subject} only has {seq_len} time points')
                        continue
                    df = df._append(sub_attr, ignore_index=True)
        df.rename(columns={self.sub_col: 'Subject_ID'}, inplace=True)
        return df

    def move_sample_files(self, target_dir):
        for _, row in self.file_df.iterrows():
            subject = row['Subject_ID']
            filename = row['Filename']
            file_dst_dir = os.path.join(target_dir, str(subject)) # w/o site
            file_src_path = os.path.join(self.path, str(subject), filename)
            os.makedirs(file_dst_dir, exist_ok=True)
            # load the data to check
            

            with open(os.devnull, 'wb') as devnull:
                subprocess.call(f'mv {file_src_path} {file_dst_dir}', shell=True, stdout=devnull, stderr=devnull)


def main(args):
    train_arch_path = os.path.join(args.src, f'ADHD200_{args.atlas}_TCs_filtfix.tar.gz')
    if not os.path.exists(train_arch_path):
        raise FileNotFoundError(f'File not found: {train_arch_path}')
    test_arch_path = os.path.join(args.src, f'ADHD200_{args.atlas}_TCs_TestRelease.tar')
    if not os.path.exists(test_arch_path):
        raise FileNotFoundError(f'File not found: {test_arch_path}')

    target_num_roi = atlas2num_roi[args.atlas.lower()]

    # dirctly save all the samples under the sample directory
    target_dir = os.path.join(args.dst, args.atlas.lower()) # lowercase to consistent with ABIDE
    os.makedirs(target_dir, exist_ok=True)
    
    # unzip the data
    print('Processing the training data...')
    temp_dir = os.path.join('./temp1')
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.devnull, 'wb') as devnull:
        subprocess.call(f'tar -xvf {train_arch_path} -C {temp_dir}', shell=True, stdout=devnull, stderr=devnull)

    sites = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f)) and 'template' not in f]
    df = pd.DataFrame()
    for site in tqdm(sites, desc='Sites', position=0):
        site_dir = os.path.join(temp_dir, site)
        site_obj = Site(site, site_dir, test=False, target_num_roi=target_num_roi)
        site_obj.move_sample_files(target_dir)
        df = pd.concat([df, site_obj.file_df], axis=0)
        df.to_csv(os.path.join(args.split, 'ADHD200_Training.csv'), index=False)
    with open(os.devnull, 'wb') as devnull:
        subprocess.call(f'rm -rf {temp_dir}', shell=True, stdout=devnull, stderr=devnull)

    print('Processing the test data...')
    temp_dir = os.path.join('./temp2')
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.devnull, 'wb') as devnull:
        subprocess.call(f'tar -xvf {test_arch_path} -C {temp_dir}', shell=True, stdout=devnull, stderr=devnull)

    sites = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f)) and 'template' not in f]
    df = pd.DataFrame()
    for site in tqdm(sites, desc='Sites', position=0):
        site_dir = os.path.join(temp_dir, site)
        site_obj = Site(site, site_dir, test=True, target_num_roi=target_num_roi)
        site_obj.move_sample_files(target_dir)
        df = pd.concat([df, site_obj.file_df], axis=0)
        df.to_csv(os.path.join(args.split, 'ADHD200_Testing.csv'), index=False)
    with open(os.devnull, 'wb') as devnull:
        subprocess.call(f'rm -rf {temp_dir}', shell=True, stdout=devnull, stderr=devnull)
    
    test_csv = pd.read_csv(args.test_csv)
    test_csv = test_csv[test_csv['DX'] != 'pending']
    test_csv['ID'] = test_csv['ID'].apply(lambda x: f'{x:07d}')
    df = df[['Filename', 'Subject_ID', 'Num_ROI', 'Seq_len']] # only keep the necessary columns
    df = pd.merge(left=df, right=test_csv, left_on='Subject_ID', right_on='ID', how='inner')
    df.to_csv(os.path.join(args.split, 'ADHD200_Testing.csv'), index=False)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='/home/r20user17/Documents')
    parser.add_argument('--dst', type=str, default='/home/r20user17/Documents/ADHD200')
    parser.add_argument('--split', type=str, default='/home/r20user17/mia/datasets/splits')
    parser.add_argument('--test_csv', type=str, default='./allSubs_testSet_phenotypic_dx.csv')
    parser.add_argument('--atlas', type=str, default='CC400')
    args = parser.parse_args()
    main(args)
