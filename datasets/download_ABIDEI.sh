#!/bin/bash

# Prompt the user for the directory to save the ABIDE I dataset
echo "Please specify the directory where you would like to save the ABIDE I dataset:"
read data_dir

# Prompt the user for the derivatives argument
echo "Please specify the derivatives you would like to fetch (e.g., 'rois_ho'):"
read derivatives_arg

# Fetch the ABIDE I dataset with specified derivatives
# Following https://github.com/Wayfear/FBNETGEN/blob/3ed6c813920966a40fca9ffca41f90a6384e333c/util/abide/01-fetch_data.py#L62
python3 -c "from nilearn import datasets; datasets.fetch_abide_pcp(data_dir='$data_dir', derivatives='$derivatives_arg', band_pass_filtering=True, global_signal_regression=False, quality_checked=False)"

# Uncomment the line below to run the registration script if needed
# python3 register_ABIDE_I.py --data_root $data_dir
