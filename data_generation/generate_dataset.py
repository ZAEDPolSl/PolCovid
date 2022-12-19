import os
import argparse
from data_generation.utils import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("source_path", help="path to raw images")
parser.add_argument("-d", "--dest", help="path to the destination folder", default='level_up')
parser.add_argument("--umap", help="whether to generate images for umap", action="store_true")
parser.add_argument("--mode", help="generate only specified images. 0 -  masks, 1 - processed images", 
                    choices=[0, 1, 2, 3], type=int, default=0)
parser.add_argument("--whole_enhance", help="whether to enhance images, default False", action="store_true")
parser.add_argument("--whole_negative", help="for negative images, default False", action="store_true")
parser.add_argument("--n_jobs", help="number of jobs to run in parallel", default=5, type=int)
args = parser.parse_args()


img_source = args.source_path
if args.dest == 'level_up':
    basename = os.path.basename(img_source) + '_data_processed'
    dest = os.path.join(os.path.dirname(img_source),basename)
else:
    dest = args.dest
whole_negative = args.whole_negative
whole_enhance = args.whole_enhance
n_jobs = args.n_jobs

img_for_unet_source = os.path.join(dest, 'tiff_for_segmentation')
segm_source = os.path.join(dest, 'png_masks')


if args.mode in [0, 1]:
    # generate images for segmentation
    print('----- Creating images for U-Net -----')
    generate_for_segmentation_images(img_source, dest, negative=False, enhance=True, n_jobs=n_jobs)
    # generate masks
    print('----- Segmentation with U-Net -----')
    generate_masks(img_for_unet_source, dest, n_jobs=n_jobs)
if args.mode == 1:
    # generate segmented and whole images
    print('----- Creating images for classification -----')
    generate_classification_images(img_source, segm_source, dest,
                                   whole_enhance=whole_enhance,
                                   whole_negative=whole_negative, n_jobs=n_jobs)
