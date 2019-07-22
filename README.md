# fmripop
A collection of python scripts and functions to (po)st(p)rocess fmridata, preprocessed with [fmriprep](https://github.com/poldracklab/fmriprep). 
The main file is `post_fmriprep.py`, which enables users to:

+ remove fmri confounds
+ filter data
+ scrub data
+ smooth data

Many of the function in this file are wrappers of [nilearn's](https://nilearn.github.io/) functions.


To learn more about the input parameters and their default values type


```
python post_fmriprep.py --help
```



#### USE CASES

Exemplary use cases are detailed below

##### CASE 0: Use all default values of parameters

```
    python remove_confounds.py --niipath /path/to/file/file_preproc.nii.gz
                               --maskpath /path/to/file/file_brainmask.nii.gz
                               --tsvpath /path/to/file/file_confounds.tsv
```

##### CASE 1: Does not regress `framwise displacement` -- used for task-fmri data

```
    python remove_confounds.py --niipath /path/to/file/file_preproc.nii.gz
                               --maskpath /path/to/file/file_brainmask.nii.gz
                               --tsvpath /path/to/file/file_confounds.tsv'
                               --low-pass None 
                               --high-pass None 
                               --fmw_disp_th None
```

##### CASE 2: Calculates scrubbing mask AND removes contaminated volumes

```
    python remove_confounds.py --niipath /path/to/file/file_preproc.nii.gz
                               --maskpath /path/to/file/file_brainmask.nii.gz
                               --tsvpath /path/to/file/file_confounds.tsv'
                               --scrubbing
                               --remove_volumes
```

##### CASE 3: Calculates scrubbing mask, but DOES NOT remove contaminated volumes

```
    python remove_confounds.py --niipath /path/to/file/file_preproc.nii.gz
                               --maskpath /path/to/file/file_brainmask.nii.gz
                               --tsvpath /path/to/file/file_confounds.tsv'
                               --low-pass None 
                               --scrubbing
```

##### CASE 4: Performs smoothing with a different width along each axis 

```
    python remove_confounds.py --niipath /path/to/file/file_preproc.nii.gz
                               --maskpath /path/to/file/file_brainmask.nii.gz
                               --tsvpath /path/to/file/file_confounds.tsv'
                               --fwhm 1.5 2.5 1.0
```
