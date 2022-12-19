import os
import numpy as np
from skimage import transform
from .IImageProcessor import IImageProcessor
from .ClassifierUtils import minmax, cut_image_to_bbox, label_lungs


class WholeImageProcessor():

    def process_image(self, norm, segm, image_size=None):
        
        mask = transform.resize(segm, (norm.shape),
                                preserve_range=True)
        mask, label_mask = label_lungs(mask)
        if len(np.unique(label_mask)) >= 3:
            final, bbox = cut_image_to_bbox(mask, norm, margin=0.05)
        else:
            final = norm
            bbox = None
        
        if image_size is not None:
            final = transform.resize(final, (image_size, image_size),
                                    preserve_range=True)
        final = minmax(final)
        return final, bbox

    def normalize(self, image, mask=None, delta=0.0025):
        dicom_array = image.copy()
        ni, mi = dicom_array.shape
        img_vec = np.reshape(dicom_array, (ni*mi, 1))
        ymax_corrected = np.quantile(img_vec, 1-delta)
        ymin_corrected = np.quantile(img_vec, delta)
        s = np.std(img_vec)
        m = np.mean(img_vec)
        ymax = m + 3*s
        ymin = m - 3*s
        p1 = np.max([ymin, ymin_corrected])
        p2 = np.min([ymax, ymax_corrected])
        dicom_array[dicom_array > p2] = p2
        dicom_array[dicom_array < p1] = p1
        dicom_norm = minmax(dicom_array)
        return dicom_norm
