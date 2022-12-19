import os
import numpy as np
from skimage import transform
from .IImageProcessor import IImageProcessor
from .ClassifierUtils import minmax, label_lungs, align_lungs, resize_to_square


class SegmentedImageProcessor():

    def process_image(self, norm, segm, image_size=None):
        
        mask = transform.resize(segm, (norm.shape),
                                preserve_range=True)
        mask, label_mask = label_lungs(mask)
        norm[mask == 0] = 0
        if len(np.unique(label_mask)) >= 3:
            limited_background = align_lungs(norm, label_mask)
            final = resize_to_square(limited_background)
        else:
            final = norm
        if image_size is not None:
            final = transform.resize(final, (image_size, image_size),
                                    preserve_range=True)
        final = minmax(final)
        return final, None

    def normalize(self, image, mask=None, delta=0.0005):
        if mask is not None:
            img_vec = image[mask > 0]
        else:
            ni, mi = image.shape
            img_vec = np.reshape(image, (ni*mi, 1))
        ymax_corrected = np.quantile(img_vec, 1-delta)
        ymin_corrected = np.quantile(img_vec, delta)
        s = np.std(img_vec)
        m = np.mean(img_vec)
        ymax = m + 3*s
        ymin = m - 3*s
        p1 = np.max([ymin, ymin_corrected])
        p2 = np.min([ymax, ymax_corrected])
        image[image > p2] = p2
        image[image < p1] = p1
        image = minmax(image)
        return image
