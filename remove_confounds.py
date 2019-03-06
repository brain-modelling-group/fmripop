#!/usr/bin/env python 

"""
This function loads three files:
    1)  *_confounds.tsv 
    2)  *_preproc.nii.gz
    3)  *_brainmask.nii.gz

It then passes the confounds timeseries and the 4D
data to a `nilearn` function which removes/regresses out
the confounds specified by the user and returns the cleaned 4D data. 

Find the latest version of this code at:
https://gist.github.com/pausz/70203386a608fcf82e5c6051054d97e1

For help type:
    python remove_confounds.py -h


Usage:
    python remove_confounds.py --niipath '/path/to/file/sub-01_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
                            --maskpath '/path/to/file/sub-01_task-rest_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'
                            --tsvpath '/path/to/file/sub-01_task-rest_bold_confounds.tsv' 


TESTED WITH:
# Anaconda 5.5.0 
# Python 3.7.0 
# pandas 0.23.4
# numpy 1.15.1
# nilearn 0.5.0
# nibabel 2.3.1

NOTES: 
Warning about using butterworth filter: https://github.com/nilearn/nilearn/issues/374

REFERENCES:  
[1] Confounds removal is based on a projection on the orthogonal of 
the signal space. Friston, K. J., A. P. Holmes, K. J. Worsley, 
J.-P. Poline, C. D. Frith, et R. S. J. Frackowiak. 
“Statistical Parametric Maps in Functional Imaging: 
A General Linear Approach”. Human Brain Mapping 2, no 4 (1994): 189-210.


[2] Lindquist, M., Geuter, S., Wager, T., & Caffo, B. (2018). 
Modular preprocessing pipelines can reintroduce artifacts into fMRI data. bioRxiv, 407676.

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

# import neuroimaging packages
import nilearn.image as nl_img



# Create parser for options
parser = argparse.ArgumentParser(
    description='Handle parameters to remove confounders from 4D frmi images.')

# These parameters must be passed to the function
parser.add_argument('--niipath', 
    type    = str, 
    default = None,
    help    ='The path to the nii.gz file whose data will be cleaned.')

parser.add_argument('--tsvpath', 
    type    = str, 
    default = None,
    help    ='The path to the tsv file with the confounders.')


parser.add_argument('--maskpath', 
    type    = str, 
    default = None,
    help    ='The path of the nii.gz file with the mask.')

# These parameters have default values. 
parser.add_argument('--nconf',
    type    = int,
    default = 9,
    help    = 'The number of confounds to be removed.')

parser.add_argument('--confound_list', 
    type    = list, 
    default = ['csf', 'white_matter', 'framewise_displacement',
               'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
    help    = 'A list with the name of the confounders to remove. Use the headers in the tsv file.')

parser.add_argument('--low_pass',
    type    = float, 
    default = 0.10,
    help    ='The low-pass filter cutoff frequency in [Hz].')

parser.add_argument('--high_pass',
    type    = float, 
    default = 0.01,
    help    ='The high-pass filter cutoff frequency in [Hz]. Set it to -1 if you dont want high-pass filtering.' )

parser.add_argument('--fmw_disp_th',
    dest    = 'fmw_disp_th', 
    type    = float, 
    default = 0.4,
    help    ='Threshold to binarize the timeseries of FramewiseDisplacement confound. This value is between 0 and 1 -- If you dont believe me, ask Luca.')

parser.add_argument('--tr',
    type    = float,
    default = 0.81,
    help    ='The repetition time (TR) in [seconds].')

parser.add_argument('--detrend', 
    dest    = 'detrend', 
    action  = 'store_true',
    default = True, 
    help    = 'Use this flag if you want to detrend the signals prior to confound removal.')

parser.add_argument('--standardize', 
    dest    = 'standardize', 
    action  = 'store_true',
    default = False,
    help    = 'Use this flag if you want to standardize the output signal between [0 1].')

parser.add_argument('--add_mean_img_back', 
    dest    = 'add_mean_img_back', 
    action  = 'store_true',
    default = True,
    help    = 'Use this flag if you want to add the mean/average original image to the cleaned data, post filtering and confound regression. Disable this flag if you do not use high-pass filtering.')


args = parser.parse_args()

# Check if we want high-pass filtering:

if args.high_pass < 0:
    args.high_pass = None
    # Disable adding the mean after cleaning the data. 
    args.add_mean_img_back = False


# This loads the tsv file in a DataFrame. 
# The function  read_csv() infers the headers of teach colum. 
# Row indexing starts at 0,
df = pd.read_csv(args.tsvpath, sep='\t')

# Remove row indices
ra = df.to_records(index=False)

tpts, nregressors = df.shape


# Check if we are removing FramewiseDisplacement
if 'FramewiseDisplacement' in args.confound_list:
    frm_disp = ra['FramewiseDisplacement']

    # Binarizer timeseries of FramewiseDisplacement
    frm_disp_bin = np.where(frm_disp > args.fmw_disp_th, 1.0, 0.0)

# Allocate memory for the confound array
confounds_signals = np.zeros((tpts, args.nconf))

# Build the confound array
for idx, this_confound in enumerate(args.confound_list):
    if this_confound != 'FramewiseDisplacement':
        confounds_signals[:, idx] = ra[this_confound]
    else:
        confounds_signals[:, idx] = frm_disp_bin

# Do the stuff
out_img = nl_img.clean_img(args.niipath, 
                                    standardize=args.standardize, 
                                    detrend=args.detrend, 
                                    confounds=confounds_signals,
                                    t_r=args.tr,
                                    high_pass=args.high_pass, 
                                    low_pass=args.low_pass,
                                    mask_img=args.maskpath)




if args.add_mean_img_back:

    # Compute the mean of the images (in the time dimension of 4th dimension)
    orig_mean_img = nl_img.mean_img(args.niipath) 

    # Add the mean image back into the clean image frames
    *xyz, time_frames = out_img.shape
    for this_frame in range(time_frames): 
        out_img.get_data()[..., this_frame] += orig_mean_img.get_data() 


# Output filename
output_filename, _ = args.niipath.split(".nii.gz") 
output_filename += '_confounds-removed.nii.gz'

# Save the clean data in a separate file
out_img.to_filename(output_filename)

# Save the input arguments in a text file with a timestamp
input_par_dict = vars(args)


timestamp = time.strftime("%Y-%m-%d-%H%M%S")
filename  = timestamp + '_input_parameters_confound_removal.txt'

with open(filename, 'w') as file:
     file.write(json.dumps(input_par_dict)) # use `json.loads` to do the reverse


print("--- %s seconds ---" % (time.time() - start_time))