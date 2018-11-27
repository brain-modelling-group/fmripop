#!/usr/bin/env python 

# author: Paula Sanz-Leon - nov 2018

# import stuff
# pandas version '0.23.4'
# Python 3.7 
# IPython 6.5.0
# Anaconda 5.5.0
# Uses nilearn & nibabel 

"""
References: 
Confounds removal is based on a projection on the orthogonal of the signal space. See Friston, K. J., A. P. Holmes, K. J. Worsley, J.-P. Poline, C. D. Frith, et R. S. J. Frackowiak. “Statistical Parametric Maps in Functional Imaging: A General Linear Approach”. Human Brain Mapping 2, no 4 (1994): 189-210.

Orthogonalization between temporal filters and confound removal is based on suggestions in Lindquist, M., Geuter, S., Wager, T., & Caffo, B. (2018). Modular preprocessing pipelines can reintroduce artifacts into fMRI data. bioRxiv, 407676.
"""
# TODO:  
# 

import pandas as pd
import numpy as np


#path_to_file = '/home/paula/mnt_ldrive/Lab_LucaC/2_OCD_TMS_Clinical_Trial/1_Patients_Records_Data/z_interim_analyses2018/x_fmriprep/1_P01_P02/1_OCD_Pre/output_dir/fmriprep/sub-01/func/'
#path_to_file = '/home/lucac/ldrive/Lab_LucaC/2_OCD_TMS_Clinical_Trial/1_Patients_Records_Data/z_interim_analyses2018/x_fmriprep/1_P01_P02/1_OCD_Pre/output_dir/fmriprep/sub-01/func/ ''

confound_filename = 'sub-01_task-rest_bold_confounds.tsv'

# This loads the file in a DataFrame. 
# The function  read_csv() infers the headers of teach colum. 
# Row indexing starts at 0,
df = pd.read_csv(confound_filename, sep='\t')


# Remove indices
ra = df.to_records(index=False)

# Get shape of the DataFrame
tpts, nregressors = df.shape

# Confounders to remove

['CSF', 'WhiteMatter','FramewiseDisplacement', 'X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']

# in [0, 1]
fw_disp_threshold = 0.4

# CSF counfound array
csf = ra['CSF']

# White matter confound array 
wm  = ra['WhiteMatter']

frm_disp = ra['FramewiseDisplacement']

frm_disp_bin = np.where(frm_disp > fw_disp_threshold, 1, 0)


# Loading the data 

from nilearn import image

fmri_filename = 'sub-01_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'


print(image.load_img('sub-01_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz').shape)


fmri_img = image.load_img('sub-01_task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
data = fmri_img.get_data()

#(x, y, t)
signals = data[:, :, 0, :]

# transpose so we have time as the first dimension / axis-0
# (t, y, x)
signals = signals.T 

# (t, y*x)
signals_2d = signals.reshape(signals.shape[0], signals.shape[1]*signals.shape[2])

detrend_flag = False

std_flag = False

tr_time = 0.81

confounds_signals = np.vstack((csf, wm, frm_disp_bin)).T

#import pdb; pdb.set_trace()

import nilearn as nl

# (t, y*x)
out_signals = nl.signal.clean(signals_2d, 
                                    standardize=std_flag, detrend=detrend_flag, 
                                    confounds=confounds_signals,
                                    t_r=tr_time,
                                    high_pass=0.01, low_pass=0.1)


# (t, y*x) -> (t, y, x) - > (x, y, t)
out_signal = np.reshape(out_signal, (signals.shape[0], signals.shape[1], signals.shape[2])).T