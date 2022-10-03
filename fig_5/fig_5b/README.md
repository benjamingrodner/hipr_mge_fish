# README for Fig 5b: HiPR-MGE-FISH on mefE gene

## Data
HiPRFISH data:
    2022_03_19_plaquephagelytic_sample_bmg_probes_non_fov_tile1_round_2_mode_spec488.czi
    2022_03_19_plaquephagelytic_sample_bmg_probes_non_fov_tile1_round_2_mode_spec514.czi
    2022_03_19_plaquephagelytic_sample_bmg_probes_non_fov_tile1_round_2_mode_spec561.czi
    2022_03_19_plaquephagelytic_sample_bmg_probes_non_fov_tile1_round_2_mode_spec633.czi

MGE-FISH data:
    20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_1_mode_airy_Airyscan_Processing_stitch.czi

## HiPRFISH processing

At the moment, the hiprfish processing was done in my main plaque experiments directory (hiprfish/plaque/experiments/2022_03_19_plaquephagelytic/data) and specific files were transferred over to the fig_5b directory

### (9/7/22) Re-running hiprfish processing from scripts only within the git repo

Documenting everything here in nb_5b_hiprfish_processing.py

Transferred hiprfish scripts into scripts/HiPRFISH

Copied probe design DSGN0673 full length probes to data/fig_5/HiPRFISH_probe_design_output

Copied the reference spectra data, fret data, to the data/fig_5/ folder.

Trained the classifier and saved the outputs to the outputs folder.

Rewrote the classifier so that there is only one function. Also debugged the umap plotting and added a thing to save the training data. Also integrated with a yaml config file. Now the output files save wherever you want, not just in the reference folder.







## MGEFISH processing

MGE processing done here with outputs folders directly in the outputs/fig5/fig5b folder.

## Manual ROI selection

The initial testing of the roi comparison method (manually outline roi in mge image, then select similar roi in hiprfish image) is recorded in the nb_roi_comparison.py. Tested [FaceMorpher](https://github.com/alyssaq/face_morpher) code (specific functions copied into ../../functions/fn_face_morpher.py) on a single roi 12. Manually selected comparable points between images. Outputs are direct into the ../../../outputs/fig_5/fig_5b folder.

Extension of the FaceMorpher idea to multiple rois is recorded in nb_roi_comparison_01.py. At this point started ../../functions/fn_manual_rois.py.
