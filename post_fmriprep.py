#!/usr/bin/env python 

"""
This module uses three files:
    1)  *_confounds.tsv 
    2)  *_preproc.nii.gz
    3)  *_brainmask.nii.gz


The confounds signals are regressed from the original fmri data. 
Optionally, the user can ask for the scrubbing mask to be calculated 
AND the contaminated volumes to be removed from the output fmri data. 

Find the latest version of this code at:
https://github.com/pausz/fmripop/blob/master/post_fmriprep.py

For help type:
    python remove_confounds.py -h


Usage:
CASE 0: Uses Default values of parameters
    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv

CASE 1: Does not regress `framwise displacement` -- used for task-fmri data
    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv'
                            --low_pass None 
                            --high_pass None 
                            --fmw_disp_th None

CASE 2: Calculates scrubbing mask AND removes contaminated volumes
    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv'
                            --calculate-scrubbing-mask
                            --remove_volumes

CASE 3: Calculates scrubbing mask, but DOES NOT remove contaminated volumes
    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv'
                            --low_pass None 
                            --calculate-scrubbing-mask

CASE 4: Performs smoothing with a different width along each axis 
    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv'
                            --fwhm 1.5 2.5 1.0

CASE 5: Remove confounds other than the default list
    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv
                            --confound_list "csf,white_matter"
                            --nconf 2


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

[3] https://www.biorxiv.org/content/biorxiv/early/2017/11/05/156380.full.pdf
    This reference recommends scrubbing BEFORE confound removal.

The reference paper, on scrubbing from which all the formulas come, is:
[4] http://www.sciencedirect.com/science/article/pii/S1053811911011815

.. moduleauthor:: Paula Sanz-Leon <paula.sanz-leon@qimrberghofer.edu.au>

TIMESTAMP: Tue 18 Aug 2020 17:13:56 AEST
"""

# import standard python packages
import argparse
from argparse import RawTextHelpFormatter
import time
import json
import os

# import standard scientific python packages
import numpy as np
import pandas as pd

# import neuroimaging packages
import nilearn.image as nl_img
import nilearn.masking as nl_mask 
import nibabel as nib

# Start tracking execution time
start_time = time.time()

# Create function to support mixed float and None type
def none_or_float(arg_value):
    if arg_value == 'None':
        return None
    return float(arg_value)

# Create class to parse numpy's ndarray
class StoreNDArray(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)

class StoreStrList(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = [item for item in values.split(',')]
        return super().__call__(parser, namespace, values, option_string)

# Create parser for options
parser = argparse.ArgumentParser(
    description='''Handle parameters to remove confounders from 4D frmi images.''',
    formatter_class = RawTextHelpFormatter)

# These parameters must be passed to the function
parser.add_argument('--niipath',
    type    = str,
    default = None,
    help    ='''The path to the nii.gz file whose data will be cleaned.''')

parser.add_argument('--tsvpath', 
    type    = str,
    default = None,
    help    ='''The path to the tsv file with the confounders.''''')

parser.add_argument('--maskpath', 
    type    = str,
    default = None,
    help    ='''The path of the nii.gz file with the mask.''')

# These parameters have default values. 
parser.add_argument('--nconf',
    type    = int,
    default = 8,
    help    = '''The number of confounds to be removed.''')

parser.add_argument('--confound_list', 
    action = StoreStrList,
    type    = str,
    #default = "csf, white_matter, trans_x, trans_y, trans_z, rot_x, rot_y, rot_z",
    default = ["csf", "white_matter", "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"],
    help    = '''A list with the name of the confounds to remove. These are headers in the tsv file.''')

parser.add_argument('--low_pass',
    type    = none_or_float, 
    default = 0.10,
    help    ='''The low-pass filter cutoff frequency in [Hz]. Set it to None if you dont want low-pass filtering.''')

parser.add_argument('--high_pass',
    type    = none_or_float, 
    default = 0.01,
    help    ='''The high-pass filter cutoff frequency in [Hz]. Set it to None if you dont want high-pass filtering.''' )

parser.add_argument('--fmw_disp_th',
    dest    = 'fmw_disp_th', 
    type    = none_or_float, 
    default = 0.4,
    help    ='''Threshold to binarize the timeseries of FramewiseDisplacement confound. This value is typically between 0 and 1 [mm]. 
             Set this flag to `None` if you do not wish to remove FramewiseDisplacement confound.''')

parser.add_argument('--tr',
    type    = float,
    default = 0.81,
    help    ='''The repetition time (TR) in [seconds].''')

parser.add_argument('--detrend', 
    dest    = 'detrend', 
    action  = 'store_true',
    default = True,
    help    = '''Use this flag if you want to detrend the signals prior to confound removal.''')

parser.add_argument('--standardize', 
    dest    = 'standardize', 
    action  = 'store_true',
    default = False,
    help    = '''Use this flag if you want to standardize the output signal between [0 1].''')

parser.add_argument('--add_mean_img_back', 
    dest    = 'add_mean_img_back', 
    action  = 'store_true',
    default = True,
    help    = '''Use this flag if you want to add the mean/average original image to the cleaned data, post filtering and confound regression. 
               Disable this flag if you do not use high-pass filtering.''')

parser.add_argument('--calculate-scrubbing-mask', 
    dest    = 'scrubbing', 
    action  = 'store_true', 
    default = False,
    help    = '''Use this flag to calculate scrubbing mask and scrubbing stats from data. 
                 Default: False
                 If True, the script does not perform volume censoring/removal. Use --remove_volumes to achieve that. ''')

parser.add_argument('--remove_volumes', 
    dest    = 'remove_volumes', 
    action  = 'store_true',
    default = False,
    help    = '''This flag determines whether contaminated volumes should be removed from the output data. Default: True.''')

parser.add_argument('--fwhm',  
   action = StoreNDArray, 
   type = float,
   nargs='+',
   default = np.array([8.0]),
   help = ''' Smoothing strength, expressed as as Full-Width at Half Maximum (fwhm), in [millimeters].
              Can be: -> a single number --fwhm 8, the width is identical along x, y and z. 
                                         --fwhm 0, no smoothing is peformed.
                      -> an array with 3 elements, --fwhm  1 1.5 2.5, giving the fwhm along each axis. ''') 

parser.add_argument('--scrub_tag',
    type    = str,
    default = '',
    help    ='''The tag appended to the output fmri nifti file whose data underwent scrubbing/volume censoring. 
                Default: empty string ''. 
                Only taken into account if scrubbing is performed''')


def fmripop_smooth_data(imgs, fwhm):
    """
    A wrapper of: 
    https://nilearn.github.io/modules/generated/nilearn.image.smooth_img.html#nilearn.image.smooth_img
    Smooth images by applying a Gaussian filter.
    args.smoothing corresponds to the parameter 'fwhm'.

    """
    out_img = nl_img.smooth_img(imgs, fwhm)
    return out_img


def fmripop_remove_confounds(args):
    """
    Removes confound signals
    """
    # label of framewise displacmeent confound as found in the tsv file
    fd_label = 'framewise_displacement'

    # This loads the tsv file in a DataFrame.
    # The function  read_csv() infers the headers of each colum.
    # Row indexing starts at 0,
    df = pd.read_csv(args.tsvpath, sep='\t')

    # Remove row indices
    ra = df.to_records(index=False)

    # Get shape of confound array
    tpts, nregressors = df.shape

    # Check if we are removing FramewiseDisplacement
    if fd_label in args.confound_list:
        frm_disp = ra[fd_label]

        # Binarizer timeseries of FramewiseDisplacement
        frm_disp_bin = np.where(frm_disp > args.fmw_disp_th, 1.0, 0.0)

    # Allocate memory for the confound array
    confounds_signals = np.zeros((tpts, args.nconf))

    # Build the confound array
    for idx, this_confound in enumerate(args.confound_list):
        if this_confound != fd_label:
            confounds_signals[:, idx] = ra[this_confound]
        else:
            confounds_signals[:, idx] = frm_disp_bin

    # Check the numeric type of the input nii image
    print("Check datatype of input nii image [header]:")
    temp_img = nib.load(args.niipath)
    # Print the datatype
    print(temp_img.header['datatype'].dtype)

    # Do the stuff
    temp_img = nl_img.clean_img(args.niipath, 
                                standardize=args.standardize, 
                                detrend=args.detrend, 
                                confounds=confounds_signals,
                                t_r=args.tr,
                                high_pass=args.high_pass, 
                                low_pass=args.low_pass,
                                mask_img=args.maskpath)

    this_dtype = np.float32
    # Add the mean image back into the clean image frames
    *xyz, time_frames = temp_img.shape
    data = np.zeros(temp_img.shape, dtype=this_dtype)

    if args.add_mean_img_back:
        # Compute the mean of the images (in the time dimension of 4th dimension)
        orig_mean_img = nl_img.mean_img(args.niipath)
        # Smooth mean image
        orig_mean_img = fmripop_smooth_data(orig_mean_img, args.fwhm[0]) # NOTE: This here is a hack because this version of nilearn does not really support a ndarray for fwhm
        for this_frame in range(time_frames):
        # Cache image data into memory and cast them into float32 
            data[..., this_frame] = temp_img.get_fdata(dtype=this_dtype)[..., this_frame] + orig_mean_img.get_fdata(dtype=this_dtype)
        
    else:
        for this_frame in range(time_frames):
        # Cache image data into memory and cast them into float32 
            data[..., this_frame] = temp_img.get_fdata(dtype=this_dtype)[..., this_frame]

    out_img = nl_img.new_img_like(temp_img, data)

    return out_img


def fmripop_calculate_scrub_mask(args):
    """
    Calculate the vector that indicates which volumes should be scrubbed/censored.
    """

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

    above_threshold = 1.0
    below_threshold = 0.0
    # Find values of framwise displacement above the threshold
    fd_bin = np.where(fd_vec > args.fmw_disp_th, above_threshold, below_threshold)

    # Frame indices around the 'contaminated frame --> index 0'
    frame_idx_before_a = -1
    frame_idx_contaminated_a = 0 # Not really used, but for completeness. fd_bin==1 at time 'a', indicates motion between frame 'a' and frame 'b''
    frame_idx_contaminated_b = 1
    frame_idx_after_b = 2

    # Create a dummy fd_bin vector padded with zeros so we can create the scrub mask with circular shifts
    fd_bin_pad = np.pad(fd_bin, (-frame_idx_before_a, frame_idx_after_b), 'constant', constant_values=(0, 0))

    # Create the scrubbing mask. This mask can be used for other stats, such as longest sequence of contaminated volumes.
    scrub_mask = (np.roll(fd_bin_pad, frame_idx_before_a) + fd_bin_pad + np.roll(fd_bin_pad, frame_idx_contaminated_b) + np.roll(fd_bin_pad, frame_idx_after_b))[-frame_idx_before_a:-frame_idx_after_b]
    
    # NOTE: Should check that for every FD above threshold detected we're removing the same number of frames
    to_delete = False
    # Create a boolean mask. A value of False indicates that the element/subarray should be deleted 
    scrub_mask = np.where(scrub_mask >= above_threshold, to_delete, not(to_delete))
    return scrub_mask


def frmipop_calculate_scrub_stats(scrub_mask, args):
    """
    This function calculates basic duration stats.
    Calculates the percentage of scrubbed volumes. 
    Calculates the length [in minutes of]:
                                         + the original data and 
                                         + the scrubbed data.

    """
    # In the boolean scub_mask False indicates that the element/subarray should be deleted 

    # Percentage of frames to be eliminated 
    tpts, *_  =  scrub_mask.shape
    scrubbed_percentage = ((1.0 - scrub_mask.sum() / tpts))*100
    stpts =  (~scrub_mask).sum()
   # Get length in minutes
    original_length = (tpts* args.tr)/60.0
    scrubbed_length = ((tpts-stpts) * args.tr)/60.0

    return scrubbed_percentage, scrubbed_length, original_length


def fmripop_remove_volumes(imgs, scrub_mask, args, this_dtype=np.float32):
    """
    Removes the 'contaminated' volumes from the 4D fmri image. 
    The resulting sequence will be shorter than the input along the 4th dimension/
    """
    # Returns a 2D array of shape timepoints x (voxels_x * voxels_y * voxels_z)
    masked_data  = nl_mask.apply_mask(imgs, args.maskpath)
    # In the boolean scub_mask False indicates that the element/subarray should be deleted 
    masked_data = masked_data[scrub_mask, :]
    out_img = nl_mask.unmask(masked_data.astype(this_dtype), args.maskpath)
    return out_img


def fmripop_scrub_data(out_img, args, params_dict):
    """
    Handle scrubbing associated actions
    """
    scrub_mask = fmripop_calculate_scrub_mask(args)
    scbper, scbl, ol = frmipop_calculate_scrub_stats(scrub_mask, args)
    params_dict['scrub_mask'] = scrub_mask.tolist() # True: uncontaminated volume. False: contaminated volume
    params_dict['original_length_min'] = ol
    params_dict['scrubbed_length_min'] = scbl
    params_dict['scrubbed_percentage'] = scbper

    if args.remove_volumes:
        out_img = fmripop_remove_volumes(out_img, scrub_mask, args)
    
    return out_img, params_dict


def fmripop_save_imgdata(args, out_img, output_tag=''):
    """
    Saves the output 4D image 
    """

    # Output filename
    output_filename, _ = args.niipath.split(".nii.gz")
    # NOTE: this line is not general enough, but it'll do for now
    output_tag = '_confounds-removed' + output_tag +  '.nii.gz'
    output_filename += output_tag 
    # Save the clean data in a separate file
    out_img.to_filename(output_filename)
    return


def fmripop_save_params(args, params_dict):
    # Save the input arguments in a text file with a timestamp
    #timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    filename = 'fmripop_parameters.json'
    jsonpath, _ = os.path.split(args.tsvpath) 

    with open(os.path.sep.join((jsonpath, filename)), 'w') as file:
        file.write(json.dumps(params_dict, indent=4, sort_keys=True)) # use `json.loads` to do the reverse
    return


def fmripop_check_args(args):
    """
    Checks input arguments and sets other dependent arguments accrodingly
    """

    # Check if we want high-pass filtering. 
    # High-pass filtering essentially removes the mean/DC component of the signal, so we need to add it back.
    if args.high_pass is None:
        # If we do not high-pass filter, disable adding the mean image back after cleaning the data.
        args.add_mean_img_back = False

    # Check if we want to regress framwise displacement
    if args.fmw_disp_th is not None:
        # Add it to the default confound list
        args.confound_list.append(fd_label)
        args.nconf += 1
    
    # If we want scrubbing check that the user has not specified a tag, otherwise set a default
    empty_str = '' 
    if args.scrubbing and (args.scrub_tag is not empty_str):
       args.scrub_tag = '_scbd'

    return args


if __name__ == '__main__':

    # Read in user-specified parameters
    args = parser.parse_args()
    # Set derived Parameters according to user specified parameters
    args = fmripop_check_args(args)
    # Convert to dict() for saving later
    params_dict = vars(args)
    params_dict['fwhm'] = args.fwhm.tolist() 

    import ipdb; ipdb.set_trace()
    # Performs main task -- removing confounds
    out_img = fmripop_remove_confounds(args)
    
    # Perform additional actions on data
    if args.scrubbing:
       out_img = fmripop_scrub_data(out_img, args, params_dict)

    if args.fwhm.sum(): # If fwhm is not zero, performs smoothing
        out_img = fmripop_smooth_data(out_img, args.fwhm) # NOTE: This here is a hack because this version  (0.5.0)of nilearn does not really support a ndarray for fwhm
    
    # Save output image and parameters used in this script
    fmripop_save_imgdata(args, out_img, output_tag=args.scrub_tag)
    fmripop_save_params(args, params_dict)

    print("--- %s seconds ---" % (time.time() - start_time))
