import os 
import imageio
import numpy as np
from PIL import Image, ImageOps
from skimage import io, img_as_float, transform
from skimage.color import rgb2gray
from tqdm import tqdm
from .ImageProcessorFactory import ImageProcessorFactory
from .ClassifierUtils import segment, convex_hull, read_dicom_image, cut_image_to_bbox
from .ClassifierUtils import load_model as custom_load_model
import pandas as pd
from joblib import Parallel, delayed

models_path = "model"


def generate_classification_images(img_source, segm_source, dest,
                                   whole_negative=False, whole_enhance=False,
                                   n_jobs=5):
    exceptions = []
    
    print('Initializing image processors for classification...', end='\r')
    whole_processor = ImageProcessorFactory(False)
    print('Initializing image processors for classification... Done.')
    
    whole_dest = os.path.join(dest, 'tiff_whole')
    
    print('Getting full paths to images...', end='\r')
    source_file_paths, dest_file_paths = get_source_dest_paths(img_source, whole_dest)
    masks_file_paths = [os.path.join(segm_source, os.path.relpath(img, img_source))\
        for img in source_file_paths]
    print('Getting full paths to images... Done.')
    
    print('Checking paths.')
    list_of_paths = list({os.path.split(path)[0] for path in dest_file_paths})

    check_path_exists(list_of_paths)
    
    print('Creating and saving images...')

    def process_image(fullname, mask_fullname, fulldest):
        file = os.path.split(fullname)[-1]
            
        try:
            ext = file.rsplit('.', 1)[-1]
            fulldest = fulldest.replace(ext, 'tiff')
            
            if not check_file_exists(fulldest):
                img = read_file(fullname)
                
                segm = Image.open(mask_fullname.replace(ext, 'png')).convert('L')
                segm = np.asarray(segm)
                
                # create whole, cropped images
                processed, _ = whole_processor.process_for_classification(img, segm, 
                                                                        enhance=whole_enhance, 
                                                                        negative=whole_negative)
                processed = img_as_float(processed)
                io.imsave(fulldest, processed)
                
                mask, bbox = cut_image_to_bbox(segm, segm, margin=0.05)
                mask = transform.resize(mask, (512, 512),
                                    preserve_range=True)
                imageio.imwrite(mask_fullname.replace(ext, 'png'),
                                (mask).astype('uint8'))
                   
        except:
            print(f'>>> Mask corrupted for image {file}')
            exceptions.append(fullname)
                
    Parallel(n_jobs=n_jobs, prefer="threads")(delayed(process_image)(input_path, input_mask, output_path)
                            for input_path, input_mask, output_path in tqdm(zip(source_file_paths,
                                                                                masks_file_paths,
                                                                                dest_file_paths),
                                                                            total=len(source_file_paths)))
    print('Done.')
    print(f'len of excepts: {len(exceptions)}')
    if len(exceptions) > 0:
        df = pd.DataFrame({'Exception': exceptions})
        df.to_csv(os.path.join(dest,'exceptions_classification_images.csv'))
    print('Images created successfully.\n')
    

def generate_for_segmentation_images(img_source,  dest,  negative=False, enhance=False, n_jobs=5):
    
    exceptions = []
    
    print('Initializing image processors...', end='\r')
    whole_processor = ImageProcessorFactory(False)
    print('Initializing image processors... Done.')
    
    if enhance:
        whole_dest = os.path.join(dest, 'tiff_for_segmentation')
    else:
        whole_dest = os.path.join(dest, 'tiff_for_umap')
    
    print('Getting full paths to images...', end='\r')
    source_file_paths, dest_file_paths = get_source_dest_paths(img_source, whole_dest)
    print('Getting full paths to images... Done.')
    
    print('Checking paths.')
    list_of_paths = list({os.path.split(path)[0] for path in dest_file_paths})
    check_path_exists(list_of_paths)
      
    print('Creating and saving images...')

    def process_image(fullname, fulldest):
        file = os.path.split(fullname)[-1]
        try:
            ext = file.rsplit('.', 1)[-1]
            fulldest = fulldest.replace(ext, 'tiff')
            
            if not check_file_exists(fulldest):
                img = read_file(fullname)
                
                processed = whole_processor.process_for_segmentation(img, 
                                                                     enhance=enhance,
                                                                     negative=negative)
                processed = img_as_float(processed)
                io.imsave(fulldest, processed)
        except:
            exceptions.append(fullname)

    Parallel(n_jobs=n_jobs, prefer="threads")(delayed(process_image)(input_path, output_path)
                            for input_path, output_path in tqdm(zip(source_file_paths, dest_file_paths),
                                                                total=len(source_file_paths)))
    print('Done.')
    
    if len(exceptions) > 0:
        df = pd.DataFrame({'Exception': exceptions})
        df.to_csv(os.path.join(dest, 'exceptions_normalized_images.csv'))
    print('Images created successfully.\n')


def generate_masks(img_source, dest, n_jobs=5):
    exceptions = []
    
    print('Loading U-Net model...', end='\r')
    segmentation_network = custom_load_model(os.path.join(
        models_path, 'unet_30102020_contrast_model.h5'))
    print('Loading U-Net model... Done.')
    
    print('Getting full paths to images...', end='\r')
    dest_masks = os.path.join(dest, 'png_masks')
    source_file_paths, dest_file_paths = get_source_dest_paths(img_source, dest_masks)
    print('Getting full paths to images... Done.')
    
    print('Checking paths.')
    list_of_paths = list({os.path.split(path)[0] for path in dest_file_paths})
    check_path_exists(list_of_paths)
   
    print('Creating and saving masks...')

    def process_image(fullname, fulldest):
        file = os.path.split(fullname)[-1]
        try:
            ext = file.rsplit('.', 1)[-1]
            fulldest = fulldest.replace(ext, 'png')
            
            if not check_file_exists(fulldest):
                img = read_file(fullname)
                
                # create segmented images
                _, mask, _ = segment(img, segmentation_network)
                mask = np.squeeze(mask)
                
                # mask, bbox = cut_image_to_bbox(mask, mask, margin=0.05)
                # mask = transform.resize(mask, (512, 512),
                #                     preserve_range=True)
                # mask = convex_hull(mask)
                imageio.imwrite(fulldest,
                                (mask.squeeze()*255).astype('uint8'))
        except:
            exceptions.append(fullname)
                
    Parallel(n_jobs=n_jobs, prefer="threads")(delayed(process_image)(input_path, output_path)
                            for input_path, output_path in tqdm(zip(source_file_paths, dest_file_paths),
                                                                total=len(source_file_paths)))
    print('Done.')
            
    if len(exceptions) > 0:
        df = pd.DataFrame({'Exception': exceptions})
        df.to_csv(os.path.join(dest, 'exceptions_masks.csv'))
    print('Masks created successfully.\n')
    
    
def read_file(fullname):
    ext = fullname.rsplit('.', 1)[-1]
    if ext.lower() == 'dcm':
        img = read_dicom_image(fullname)
    elif ext == 'tiff' or ext == 'png':
        img = io.imread(fullname)
    else:
        img = Image.open(fullname).convert('L')
        img = np.asarray(img)
    if len(img.shape) > 2:
        img = rgb2gray(img)
    return img


def check_file_exists(filename):
    return os.path.isfile(filename)


def get_source_dest_paths(source, dest):
    source_file_paths = []
    dest_file_paths = []
    for path, subdirs, files in os.walk(source):
        for name in files:
            file_fullpath = os.path.join(path, name)
            spec = os.path.relpath(file_fullpath, source)
            subdirs = os.path.dirname(spec)
            source_file_paths.append(file_fullpath)
            dest_file_paths.append(os.path.join(dest, subdirs, name))
    return source_file_paths, dest_file_paths


def check_path_exists(list_of_paths):
    for path in list_of_paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print('Destination directory created.')
        else:
            print('Destination directory arleady exists.')