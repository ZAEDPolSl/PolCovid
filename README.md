# PolCovid
This repository is intended for generating processed images and masks as described in the study
"POLCOVID: a multicenter multiclass chest X-ray database (Poland, 2020-2021)", A. Suwalska, J. Tobiasz, W. Prazuch, M. Socha et al.. 
The pipeline was used to generate PolCovid processed images 
that are shared with the study.

To generate processed images and masks from original chest X-Ray images, run:

`python -m data_generation.generate_dataset path_to_folder_with_images --mode 1`

The first argument to the function is a path to folder with original images. 

The pipeline works for images in formats: .dcm, .jpg, .jpeg, .png, .tiff. 
