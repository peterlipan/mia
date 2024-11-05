#!/bin/bash

echo "Please specify the directory where you would like to save the ADHD 200 dataset:"
read data_dir

python3 -c "from nilearn import datasets; datasets.fetch_adhd(n_subjects=40, data_dir='$data_dir')"
# python3 register_ABIDE_I.py --data_root $data_dir
