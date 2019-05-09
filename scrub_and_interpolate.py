#!/usr/bin/env python 

"""
INPUT: This function loads three files:
    1)  *_confounds.tsv 
    2)  *_preproc.nii.gz   --> This file can be the one produced by fmriprep or the one with the confounds removed
    3)  *_brainmask.nii.gz 

The function uses the confound 'framewise_displacment' from the *.tsv file to 
determine which frames should be scrubbed.

OUTPUT: This function outputs two main files.
        1)  *_confounds_XXX.tsv or 
        2)  *_preproc_XXX.nii.gz 

where XXX is a three letter string 'scb' (scrubbed) and/or 'ipd' (interpolated).
If the input files are scrubbed and interpolated then the output files will have
'_scb_ipd' appended to the original filename. If only scrubbing is performed, then 
'_scb' is appended to the original filename.


For help type:
    python scrub_and_interpolate.py -h


Usage:
python scrub_and_interpolate.py --niipath '/path/to/file/sub-XX_preproc.nii.gz'
                                --maskpath '/path/to/file/sub-01_task-rest_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'
                                --tsvpath '/path/to/file/sub-01_task-rest_bold_confounds.tsv' 


TESTED WITH:
# Anaconda 5.5.0 
# Python 3.7.0 
# pandas 0.23.4
# numpy 1.15.1
# scipy XXXX

CODE REFERENCES:  
https://github.com/SIMEXP/niak/blob/master/commands/SI_processing/niak_fd2mask.m
https://github.com/FCP-INDI/C-PAC/blob/7965125125319d895464f40db57b0eaba9c84da3/CPAC/generate_motion_statistics/generate_motion_statistics.py

Nilearn: Typically, the FD is padded back (ie a value 0 is added at the back) 
and thresholded and 1 TR is removed before, and 2 after (a total of 4 frames).

NOTE: however, fmriprep however prepends a nan to the FD vector, in which case 
one should remove the frame that aligns with the fd 'frame' ==1 (eg, FA), the 
frame before(FA-1), the frame before that one (FA-2), and one frame after
(FA+1). So in that case it would be remove '2 frames before and 1 frame after' 

# NOTE: TODO: the despike optimised method of scrubbing, removes data before 
removal of confounds and also interpolates the confound signals.


BIB REFERENCES:
[1] https://www.biorxiv.org/content/biorxiv/early/2017/11/05/156380.full.pdf
    This reference recommends scrubbing BEFORE confound removal.

The reference paper, on scrubbing from which all the formulas come, is:
[2] http://www.sciencedirect.com/science/article/pii/S1053811911011815

.. moduleauthor:: Paula Sanz-Leon <paula.sanz-leon@qimrberghofer.edu.au>
    
"""

# import standard python packages
import argparse
import time
import json

# Start tracking execution time
start_time = time.time()

# import standard scientific python packages
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

import nibabel as nib
from nilearn.masking import apply_mask, unmask
import nilearn.image as nl_img

# Create function to support mixed float and None type
def none_or_float(arg_value):
    if arg_value == 'None':
        return None
    return arg_value

# Create parser for options
parser = argparse.ArgumentParser(
    description='Handle parameters to scrub and interpolate 4D fmri images.')

# These parameters must be passed to the function
parser.add_argument('--niipath', 
    type    = str, 
    default = None,
    help    ='The path to the nii.gz file whose data will be scrubbed.')

parser.add_argument('--tsvpath', 
    type    = str, 
    default = None,
    help    ='The path to the tsv file with the confounds.')

parser.add_argument('--maskpath', 
    type    = str, 
    default = None,
    help    ='The path of the nii.gz file with the mask.')

parser.add_argument('--nb_vol_min', 
    type    = int, 
    default = 128,
    help    = 'An integer with the minimum number of time frames that should be preserved. NOTE: Not used at the moment.')

parser.add_argument('--fmw_disp_th',
    dest    = 'fmw_disp_th', 
    type    = float, 
    default = 0.4,
    help    ='Framewise displacment threshold in [mm]. This value is used to determine which volumes will be scrubbed.')

parser.add_argument('--tr',
    type    = float,
    default = 0.81,
    help    ='The repetition time (TR) in [seconds]. NOTE: Not used at the moment, but may be useful later')

parser.add_argument('--interpolate', 
    dest    = 'do_interpolation', 
    action  = 'store_true',
    default = True,
    help    = 'Use this flag to define if scrubbed data are interpolated or not. Default: True')

def visual_debug(time_vec, data, voxel_to_plot=42, fname='none'):
    plt.plot(time_vec, data[:, voxel_to_plot])
    plt.savefig(fname)
    return
    
args = parser.parse_args()
output_tag = '_scb'

# This loads the tsv file in a DataFrame. 
# The function  read_csv() infers the headers of teach colum. 
# Row indexing starts at 0,
df = pd.read_csv(args.tsvpath, sep='\t')

# Remove row indices
ra = df.to_records(index=False)
# Get original nummber of frames
tpts, _ = df.shape

# Get framewise displacement vector
fd_vec = ra['framewise_displacement']

# fmriprep *prepends* nan in the FD vector, other approaches *append* a 0 at the end of the vector
# here we shift the fd to match 'typical' approaches in the literature
roll_step = -1
fd_vec = np.roll(fd_vec, roll_step) 
fd_vec[tpts-1] = 0

# Find values of framwise displacement above the threshold
above_threshold = 1.0
below_threshold = 0.0
fd_bin = np.where(fd_vec > args.fmw_disp_th, above_threshold, below_threshold)

nb_frames = len(fd_bin) # This should be == tpts-1

frame_idx_before_a = -1
frame_idx_contaminated_a = 0 # Not really used, but for completeness. fd_bin==1 at time 'a', indicates motion between frame 'a' and frame 'b''
frame_idx_contaminated_b = 1
frame_idx_after_b  =  2

# Create a dummy fd_bin vector padded with zeros so we can create the scrub mask with circular shifts
fd_bin_pad = np.pad(fd_bin, (-frame_idx_before_a, frame_idx_after_b), 'constant', constant_values=(0, 0))

# Create the scrubbing mask 
scrub_mask = (np.roll(fd_bin_pad, frame_idx_before_a) + fd_bin_pad + np.roll(fd_bin_pad, frame_idx_contaminated_b) + np.roll(fd_bin_pad, frame_idx_after_b))[-frame_idx_before_a:-frame_idx_after_b]

# NOTE: Should check that for every FD above threshold detected we're removing the same number of frames
to_delete = False
# Create a boolean mask. A value of False indicates that the element/subarray should be deleted 
scrub_mask = np.where(scrub_mask >= above_threshold, to_delete, not(to_delete))

# Percentage of frames to be eliminated 
scrubbed_percentage = (1.0 - scrub_mask.sum() / nb_frames)
# Check the numeric type of the input nii image
print("Percentage of scrubbed frames:")
# Print the datatype
print(scrubbed_percentage*100)

# Returns a 2D array of shape timepoints x (voxels_x * voxels_y * voxels_z)
masked_data  = apply_mask(args.niipath, args.maskpath)
tpts, voxels = masked_data.shape
time_vec = np.arange(0, tpts) * args.tr
time_vec_scrubbed = time_vec[scrub_mask]
# Get length in minutes
original_length = (tpts * args.tr)/60.0
scrubbed_length = (time_vec_scrubbed.shape * args.tr)/60.0

# original data
visual_debug(time_vec, masked_data, fname='original')

def scrub_data(frmi_data, confound_data, scrub_mask):
    """
    fmri_data: ndarray
    confound_data: pandas dataframe 
    """
+
    # Remove bad frames
    masked_data = masked_data[scrub_mask, :]
    # HERE - remove rows of confounds
    confound_data.drop(scrub_mask, axis=0, inplace=True). 


# Scrubbed data
visual_debug(time_vec_scrubbed, masked_data, fname='scrubbed')

############################# Interpolate data ###############################

# Reshape the data into a 4D arraythis_dtype = np.float32
this_dtype = np.float32
out_img = unmask(masked_data.astype(this_dtype), args.maskpath)
visual_debug(time_vec, masked_data, fname='interpolated')

# Output filename
output_filename, _ = args.niipath.split(".nii.gz") 
output_filename += output_tag
output_filename += '.nii.gz'

# Save the scrubbed (and interpolated) data in a separate file
out_img.to_filename(output_filename)

# Save the input arguments in a text file with a timestamp
input_par_dict = vars(args)
input_par_dict['orginal_length_min'] = original_length
input_par_dict['scrubbed_length_min'] = scrubbed_length
input_par_dict['scrubbed_percentage'] = scrubbed_percentage*100

timestamp = time.strftime("%Y-%m-%d-%H%M%S")
filename  = timestamp + '_io_parameters_scrub_interp.txt'

with open(filename, 'w') as file:
     file.write(json.dumps(input_par_dict)) # use `json.loads` to do the reverse

print("--- %s seconds ---" % (time.time() - start_time))
