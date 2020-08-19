

[![DOI](https://zenodo.org/badge/186916000.svg)](https://zenodo.org/badge/latestdoi/186916000)


# fmripop
A collection of python scripts and functions to (po)st(p)rocess fmridata, preprocessed with [fmriprep](https://github.com/poldracklab/fmriprep). 
The main file is `post_fmriprep.py`, which enables users to:

+ remove fmri confounds
+ filter data
+ smooth data
+ scrub data
+ censor volumes

Many of the function in this file are wrappers of [nilearn's](https://nilearn.github.io/) functions.


To learn more about the input parameters and their default values type


```
python post_fmriprep.py --help
```

#### USE CASES

Exemplary use cases are detailed below

##### CASE 0-a: Default values, typical use for *resting-state fmri data*
Uses default values of non-boolean parameters (ie, confounds, filtering, tr)
and outputs data centred around a nonzero mean. The default values of these
non-boolean parameters are optimised for resting-state fmri data.

```
        python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv
                            --detrend
                            --add_orig_mean_img
```


##### CASE 0-b: Default values, typical use for resting state fmri data, zero-centred data.
Uses default values of non-boolean parameters (ie, confounds, filtering, tr)
and outputs zero-centred data (mean==0).

    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv
                            --detrend


##### CASE 1: Typical use case for *task fmri data*.
        Does not regress `framwise displacement` -- used for task-fmri data
        Does not filter.
        Uses default value for smoothing the data

```
    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv'
                            --detrend
                            --add_orig_mean_img
                            --low_pass None 
                            --high_pass None 
                            --fmw_disp_th None
```

##### CASE 2: Calculates scrubbing mask AND removes contaminated volumes

```
    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv'
                            --detrend
                            --add_orig_mean_img
                            --calculate_scrubbing_mask
                            --remove_volumes
```

##### CASE 3: Calculates scrubbing mask, but DOES NOT remove contaminated volumes

```
    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv'
                            --detrend
                            --add_orig_mean_img
                            --calculate_scrubbing_mask
```

##### CASE 4: Performs smoothing with a different width along each axis 

```
    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv'
                            --detrend
                            --add_orig_mean_img
                            --fwhm 1.5 2.5 1.0
```

##### CASE 5: Remove confounds other than those in the default list

```
    python post_fmriprep.py --niipath /path/to/file/file_preproc.nii.gz
                            --maskpath /path/to/file/file_brainmask.nii.gz
                            --tsvpath /path/to/file/file_confounds.tsv
                            --detrend
                            --add_orig_mean_img
                            --confound_list "csf,white_matter"
                            --num_confounds 2
```