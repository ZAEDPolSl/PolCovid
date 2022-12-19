import os
import numpy as np
from skimage import transform, io, img_as_float
from .WholeImageProcessor import WholeImageProcessor
from .SegmentedImageProcessor import SegmentedImageProcessor
from .ClassifierUtils import get_temp_image_size, convex_hull, CE_e, read_dicom_image
import cv2


class ImageProcessorFactory():

    def __init__(self, are_lungs):

        if are_lungs:
            self._processing_strategy = SegmentedImageProcessor()
        else:
            self._processing_strategy = WholeImageProcessor()

    def process_for_classification(self, image, mask, enhance=False, negative=False, image_size=512):
        # convex hull for mask
        mask_hull = convex_hull(mask)
        # resize image and mask
        image, mask = self._resize_inputs(image, mask_hull)
        # normalize image
        image = self._processing_strategy.normalize(image, mask=mask)
        # transform inputs
        image = self._transform_image(image, enhance=enhance, negative=negative)
        # process inputs
        processed, bbox = self._processing_strategy.process_image(image, mask, image_size=image_size)
        # get sizes of images
        if bbox is not None:
            stats = [bbox.shape, np.sum(bbox)]
        else:
            stats = [1, 0]
        return processed, stats

    def process_for_segmentation(self, image, enhance=False, negative=False):
        # resize image and mask
        image = self._resize_inputs(image)
        # normalize image
        image = self._processing_strategy.normalize(image, mask=None)
        # transform inputs
        image = self._transform_image(image, enhance=enhance, negative=negative)
        # resize image
        norm = transform.resize(image, (512, 512), preserve_range=True)
        return norm

    @staticmethod
    def _resize_inputs(image, mask=None):
        temp_image_size = get_temp_image_size(image)
        image = transform.resize(image, (temp_image_size, temp_image_size),
                                 preserve_range=True)
        if mask is not None:
            mask = transform.resize(mask, (temp_image_size, temp_image_size),
                                    preserve_range=True)
            return image, mask
        return image

    @staticmethod
    def _transform_image(image, enhance=False, negative=False):
        # Contrast enhancement for dicom images
        if enhance:
            image = CE_e(image)
        # Flip if negative
        if negative:
            image = image.max() - image
        return image

    @staticmethod
    def _save_image(image, ext='tiff'):
        image = img_as_float(image)
        io.imsave('temp.'+ext, image)
