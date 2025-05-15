import os
import cv2
import albumentations as A

from abc import ABC, abstractmethod
from typing import List

class AugmentorBase(ABC):
    def __init__(self, data_dir: str, augmentation_type: str = "both"):
        self.data_dir = data_dir
        self.augmentation_pipeline = self._build_pipeline(augmentation_type)

    def _build_pipeline(self, augmentation_type: str) -> A.Compose:
        geometric_pipeline = [
            A.Flip(p=1),
            A.ElasticTransform(p=0.3),
            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        photometric_pipeline = [
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
            A.GaussianBlur(blur_limit=(1, 3), p=0.5),
        ]
        if augmentation_type == 'geometric':
            return A.Compose(geometric_pipeline)
        elif augmentation_type == 'photometric':
            return A.Compose(photometric_pipeline)
        return A.Compose(geometric_pipeline + photometric_pipeline)

    @abstractmethod
    def run(self):
        pass

    def _augment_image(image_path: str, n_augment: int, pipeline: A.BasicTransform) -> List:
        image = cv2.imread(image_path)
        return [pipeline(image=image)['image'] for _ in range(n_augment)]

    def _save_augmented_images(self, images: List, output_dir: str, base_idx: int):
        for j, augmented_image in enumerate(images):
            output_path = os.path.join(output_dir, f"aug_{base_idx}_{j}.png")
            cv2.imwrite(output_path, augmented_image)