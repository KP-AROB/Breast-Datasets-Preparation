import numpy as np
import cv2

def clahe(img, clip=4.0): #1.5
    """
    Image enhancement using CLAHE. Converts float image to uint8.
    """
    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)
    img = (img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(18,18))
    cl = clahe.apply(img)
    return cl

# Gabor filters to extract features
def apply_gabor(image, theta=0):
  kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
  filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
  return filtered

def minmax_normalisation(img, rangemax = 255): #1.5
    """
    Image enhancement using min-max normalization.
    """
    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)
    else:
        print("Warning: Image.max() = 0, returning uncorrect array.")
    img = (img * rangemax)
    return img



def truncate_normalization(image_mask: tuple):
    """Normalize an image within a given ROI mask

    Args:
        image_mask (tuple): tuple containing cropped images and roi masks

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
