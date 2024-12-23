#!/bin/bash

# Prompt the user for the directory to save the ABIDE I dataset
echo "Please specify the directory where you would like to save the ABIDE I dataset:"
read data_dir

# Prompt the user for the derivatives argument
echo "Please specify the derivatives you would like to fetch (e.g., 'rois_ho'):"
read derivatives_arg

# Fetch the ABIDE I dataset with specified derivatives
python3 -c "from nilearn import datasets; datasets.fetch_abide_pcp(data_dir='$data_dir', derivatives='$derivatives_arg')"

# Uncomment the line below to run the registration script if needed
# python3 register_ABIDE_I.py --data_root $data_dir
