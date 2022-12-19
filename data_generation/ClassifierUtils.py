import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from skimage.color import rgb2gray
from PIL import Image
import tensorflow as tf
from deepctr.layers import custom_objects
from tensorflow.keras import backend
from tensorflow.keras.layers import Flatten
from skimage.morphology import convex_hull_image
from scipy.ndimage import gaussian_gradient_magnitude, morphology
from skimage.morphology import closing, opening, selem
from skimage import transform, measure, io, img_as_float
from scipy.ndimage.morphology import binary_fill_holes
import SimpleITK as sitk
from scipy import stats
from collections import OrderedDict
import skimage.morphology as morph
import pydicom
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def separate_lungs(image):
    """ Separates lungs after processing with align_lungs
    """
    mask = np.where(image > 0, 1, 0)

    mask = morphology.binary_erosion(mask, np.ones((5, 5)))
    mask = morphology.binary_fill_holes(mask)

    labeled_mask = measure.label(mask)
    props = measure.regionprops(labeled_mask)
    if len(props) < 2:
        tmp = np.sum(mask, axis=0)
        tmp[0:np.round(0.4*mask.shape[1]).astype('int')] = 1000
        tmp[np.round(0.6*mask.shape[1]).astype('int'):] = 1000
        min_val = np.min(tmp)
        min_ind = list(np.where(tmp == min_val))[0][0]
        if min_val > 0:
            mask[:, min_ind] = 0
    return np.where(mask, 1, 0)


def read_dicom_image(dicom_img_path):
    """Loads dicom data."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_img_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    dicom_data = reader.Execute()
    dicom_array = sitk.GetArrayFromImage(dicom_data)
    if len(dicom_array.shape) > 2:
        dicom_array = np.moveaxis(dicom_array, 0, -1)  # because in sitk convention Axial is moved to the front
        dicom_array = np.squeeze(dicom_array)  # remove redundant axis
    try:
        if is_negative(dicom_data):
            dicom_array = dicom_array.max() - dicom_array
    except:
        pass
    return dicom_array


def minmax(img):
    return (img - np.min(img))/(np.max(img)-np.min(img) + 1e-10)


def is_negative(reader):
    """Check if dicom is negative - version for sitk"""
    negative_tags = {
        '2050|0020': 'INVERSE',
        '0028|0004': 'MONOCHROME1'
    }
    for tag, negative_tag_value in negative_tags.items():
        try:
            tag_value = reader.GetMetaData(tag)
        except:
            continue
        if tag_value == negative_tag_value:
            return True
    return False


def get_temp_image_size(image, dest_size=1024):
    """ Checks the input image size to determine the appropriate
    size during processing.
    """
    n, m = image.shape
    if n > dest_size or m > dest_size:
        return dest_size
    else:
        return np.max([n, m])


def get_ordered_labels(label_mask):
    """ Determines the lung labels in labeled mask from left to right.
    """
    temp_vec = np.reshape(label_mask, (label_mask.shape[0]*label_mask.shape[1], 1),
                          order='F')
    temp_list = list(temp_vec[temp_vec > 0])
    labels = list(OrderedDict.fromkeys(temp_list))
    return labels


def resize_to_square(image):
    """ Resizes and padds numpy array to destination size maintaining the
    aspect ratio.
    """
    x_temp, y_temp = image.shape
    if x_temp != y_temp:
        if x_temp < y_temp:
            target_size = y_temp
            border_size = np.round((target_size - x_temp)/2).astype('int')
            new_image = np.zeros((target_size, target_size))
            new_image[border_size:border_size+x_temp, :] = image
        else:
            target_size = x_temp
            border_size = np.round((target_size - y_temp)/2).astype('int')
            new_image = np.zeros((target_size, target_size))
            new_image[:, border_size:border_size+y_temp] = image
    else:
        new_image = image
    return new_image


def label_lungs(mask):
    """ Takes two largest objects in the mask and label them.
    """
    mask = np.where(mask > 0, 1, 0)
    mask = morphology.binary_fill_holes(mask, structure=np.ones((5, 5))).astype(int)
    label_mask = measure.label(mask)
    props = measure.regionprops(label_mask)
    areas = [props[x].area for x in range(len(props))]
    M = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
    M = [i+1 for i in M]
    new_label_mask = label_mask * 0
    if len(M) >= 2:
        new_label_mask[label_mask == M[0]] = 1
        new_label_mask[label_mask == M[1]] = 2
    return mask, new_label_mask


def align_lungs(dicom_norm, label_mask):
    ''' Lungs alignment procedure.
    '''
    lung_labels = get_ordered_labels(label_mask)

    m1 = np.zeros(label_mask.shape)
    m2 = np.zeros(label_mask.shape)

    m1[label_mask == lung_labels[0]] = 1
    m2[label_mask == lung_labels[1]] = 1

    cut_m1, _ = cut_image_to_bbox(m1, dicom_norm)
    cut_m2, _ = cut_image_to_bbox(m2, dicom_norm)
    x1, y1 = cut_m1.shape
    x2, y2 = cut_m2.shape

    limited = np.zeros((np.max([x1, x2]), y1+y2))

    limited[0:x1, 0:y1] = cut_m1
    limited[0:x2, y1:y1+y2] = cut_m2
    return limited


def add_margin_to_bbox(image, bbox, perc=0.05):
    ''' Adds margin of specified percent of pixels to the bounding box.
    '''
    n, m = image.squeeze().shape
    bbox_n = bbox[2] - bbox[0]
    bbox_m = bbox[3] - bbox[1]
    x_mar = bbox_n * perc
    y_mar = bbox_m * perc
    new_b0 = np.round(np.max([0, bbox[0]-x_mar])).astype('int')
    new_b1 = np.round(np.max([0, bbox[1]-y_mar])).astype('int')
    new_b2 = np.round(np.min([n, bbox[2]+x_mar])).astype('int')
    new_b3 = np.round(np.min([m, bbox[3]+y_mar])).astype('int')
    return new_b0, new_b1, new_b2, new_b3


def cut_image_to_bbox(binary, dicom_norm, margin=None):
    ''' Cuts image to the edges of an object in binary mask.
    '''
    props = measure.regionprops(binary.astype('int'))
    bbox = props[0].bbox
    if margin is not None:
        b0, b1, b2, b3 = add_margin_to_bbox(dicom_norm, bbox, perc=margin)
    else:
        b0, b1, b2, b3 = bbox
    cut_image = dicom_norm[b0:b2, b1:b3]
    cut_mask = binary[b0:b2, b1:b3]
    return cut_image, cut_mask


def CE_e(img, radius=6, stop_condition=20):
    '''
    Contrast Enhancement of Medical X-Ray ImageUsing Morphological
    Operators with OptimalStructuring Element, https://arxiv.org/pdf/1905.08545.pdf
    :param img: 2D np array, image
    :param radius: int [1-N], radius of the structuring element used for morphology operations
    :param stop_condition: int, value to which Edge content (EC) difference is compared,
                           if EC difference is smaller
    then 'stop_condition' current value of radius consider to be optimal
                          (recommended: 10-100 depending on the problem)
    :return: 2D np array, Contrast enhanced image normalized in between values [0-1]
    '''
    A = minmax(img)
    ECA = np.sum(gaussian_gradient_magnitude(A, 1))
    prevEC = 0
    # radius adapt to the image
    if radius is None:
        convMtx = [np.Inf]
        for r in range(1, 15):
            # define SE as B
            B = selem.disk(r)
            # opening and closing operations defined in the paper
            Atop = A - opening(A, selem=B)
            Abot = closing(A, selem=B) - A
            Aenhanced = A + Atop - Abot
            Aenhanced = np.clip(Aenhanced, a_min=0, a_max=1)
            # Edge content calculations
            EC = np.sum(gaussian_gradient_magnitude(Aenhanced, 1))
            # min max scaling processed image
            # Aenhanced_normed = (Aenhanced - np.min(Aenhanced))/(np.max(Aenhanced)-np.min(Aenhanced))
            # stopping condition
            conv = EC - prevEC
            convMtx.append(conv)
            if convMtx[-2]-convMtx[-1] < stop_condition:
                break
            prevEC = EC
    # pre-defined radius
    else:
        B = selem.disk(radius)
        Atop = A - opening(A, selem=B)
        Abot = closing(A, selem=B) - A
        Aenhanced = A + Atop - Abot
        Aenhanced = np.clip(Aenhanced, a_min=0, a_max=1)

        EC = np.sum(gaussian_gradient_magnitude(Aenhanced, 1))

        # Aenhanced_normed = (Aenhanced - np.min(Aenhanced)) / (np.max(Aenhanced) - np.min(Aenhanced))
    return Aenhanced


def convex_hull(image, set_constant=1):
    '''
    Convex hull, create set of pixels included into the smallest convex polygon that surround all white pixels in the
    input.
    Args:
        image: [np.array] binary mask image (object of interest has to be white)
        set_constant: [int] value with which the mask is going to be filled
    Returns:
        {np.array} resulting polygons representing lungs included in the input image
    '''
    if np.sum(image) > 0:
        mask, new_label_mask = label_lungs(image)
        # if label lungs failed because there is only one lung
        if len(np.unique(new_label_mask)) < 2:
            image_chull = convex_hull_image(mask)
        # if there are two lungs
        else:
            # get left and tight half
            lung1 = np.zeros(image.shape)
            lung2 = np.zeros(image.shape)
            lung1[new_label_mask == 1] = 1
            lung2[new_label_mask == 2] = 1
            # convex hull
            chull_lung1 = convex_hull_image(lung1)
            chull_lung2 = convex_hull_image(lung2)
            # form final image
            image_chull = chull_lung1 + chull_lung2
        image_chull[image_chull > 0] = set_constant
        return image_chull
    return image


def fix_dims(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
    return image


def segment(image, model):
    # image = resize_pad_image(image)
    image_original = image.copy()
    image = preprocess(image)
    mask = model.predict(image)
    mask, reg = postprocess(mask)
    image_segmented = extract_lungs(image_original, mask[0, :, :, 0])  #
    return image_segmented, mask, reg


def preprocess(image):
    image = standardize(image)
    image = fix_dims(image)
    return image


def postprocess(mask):
    mask = mask.astype('uint8')
    # mask[0, :, :, 0] = morphology.binary_closing(mask[0, :, :, 0])
    mask[0, :, :, 0], reg = retain_big_objects(mask[0, :, :, 0], return_regions=True)
    mask[0, :, :, 0] = smoothen(mask[0, :, :, 0])
    mask[0, :, :, 0] = binarize(mask[0, :, :, 0])
    return mask, reg


def extract_lungs(image, mask, save_path=None):
    new_image = image * mask
    if save_path is not None:
        cv2.imwrite(save_path, new_image)
    return new_image

def dice_coef(y_true, y_pred):
    flatten_layer = Flatten()
    y_true_f = flatten_layer(y_true)
    y_pred_f = flatten_layer(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def standardize(X, mean=0.5802127, st_deviation=0.288578):
    X = (X - mean) / st_deviation
    return X


def resize_pad_image(im, dest_size=512, mask=False):
    old_size = im.size

    ratio = float(dest_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (dest_size, dest_size))
    new_im.paste(im, ((dest_size - new_size[0]) // 2,
                      (dest_size - new_size[1]) // 2))

    new_im = new_im.convert('L')

    if mask:
        new_im = new_im.point(lambda x: 0 if x < 128 else 255, '1')

    return np.array(new_im)


def binarize(Ypred):
    Ypred[Ypred > 0.5] = 1
    Ypred[Ypred < 0.5] = 0
    return Ypred


def retain_big_objects(mask, number=2, return_regions=False):
    labels_mask = measure.label(mask)
    regions = measure.regionprops(labels_mask)
    if len(regions) > 1:
        regions.sort(key=lambda x: x.area, reverse=True)
        for rg in regions[number:]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
        regions = regions[:number]
    labels_mask[labels_mask != 0] = 1

    if return_regions:
        return labels_mask, regions
    return labels_mask


def smoothen(image):
    im_arr_morph = image.copy()
    im_arr_morph = morph.area_opening(im_arr_morph)
    im_arr_morph = morph.area_closing(im_arr_morph)
    # im_hull = morph.convex_hull_object(im_arr_morph)
    im_arr_morph = binary_fill_holes(im_arr_morph)
    return im_arr_morph


def load_model(model_path):
    custom_objects["dice_coef"] = dice_coef
    custom_objects["dice_coef_loss"] = dice_coef_loss
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model


def calculate_shape_stats(reg):
    major_axis = np.array([x.major_axis_length for x in reg])
    perim = np.array([x.perimeter for x in reg])
    area = np.array([x.area for x in reg])
    C1 = np.mean(perim) / np.pi
    C2 = 2 * np.sqrt(np.mean(area) / np.pi)
    Lp2 = np.mean(major_axis) / np.mean(perim)
    if np.isnan(Lp2) or np.isnan(C2):
        Lp2 = 0
        C2 = 0
    return C1, C2, Lp2


def score_lungs(masks):
    reg = measure.label(masks)
    reg = measure.regionprops(reg)
    C1 = []
    C2 = []
    Lp2 = []
    if len(reg) > 0:
        C1, C2, Lp2 = calculate_shape_stats(reg)
    else:
        C1, C2, Lp2 = 0, 0, 0
    return C1, C2, Lp2


def check_input_image(segm, scaler, svm):
    C1, C2, Lp2 = score_lungs(segm)
    X = np.transpose([C1, C2, Lp2])
    X = np.expand_dims(X, axis=0)
    X_scaled = scaler.transform(X)
    y_pred = svm.predict(X_scaled)
    return np.where(y_pred == 1, True, False)
