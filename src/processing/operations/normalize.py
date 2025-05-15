import numpy as np
import cv2

def clahe(img, clip=1.5):
    """
    Image enhancement.
    @img : numpy array image
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(img)
    return cl

def truncate_normalization(image_mask: tuple):
    """Normalize an image within a given ROI mask

    Args:
        source (list): list of tuples containing cropped images and roi masks

    Returns:
        np.array: normalized image
    """
    img, mask = image_mask
    Pmin = np.percentile(img[mask != 0], 2)
    Pmax = np.percentile(img[mask != 0], 99)
    truncated = np.clip(img, Pmin, Pmax)
    if Pmax != Pmin:
        normalized = (truncated - Pmin) / (Pmax - Pmin)
    else:
        normalized = np.zeros_like(truncated)
    normalized[mask == 0] = 0
    return normalized
