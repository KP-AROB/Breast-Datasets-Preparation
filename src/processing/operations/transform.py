import cv2
import numpy as np
from .normalize import clahe, truncate_normalization


def crop_to_roi(image: np.array):
    """Crop mammogram to breast region.

    Args:
        img_list (list): List of original image as uint8 np.arrays

    Returns:
        tuple (list, list): (cropped_images, rois)
    """
    original_image = image[100:-100, 100:-100]
    image = clahe(original_image, 1.0)

    _, breast_mask = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(
        breast_mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    norm_image = truncate_normalization((original_image[y: y + h, x: x + w], breast_mask[y: y + h, x: x + w]))
    return norm_image


def resize(image: np.array, new_size=2012):
    try:
        return cv2.resize(
            image,
            (new_size, new_size),
            interpolation=cv2.INTER_LINEAR,
        )
    except Exception as e:
        return None
