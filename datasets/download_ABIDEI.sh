#!/bin/bash

echo "Please specify the directory where you would like to save the ABIDE I dataset:"
read data_dir

python3 -c "from nilearn import datasets; datasets.fetch_abide_pcp(data_dir='$data_dir', derivatives='rois_cc400')"
# python3 register_ABIDE_I.py --data_root $data_dir