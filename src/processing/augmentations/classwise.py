import os
import logging
from glob import glob
from tqdm import tqdm
from typing import List
from .base import AugmentorBase

class ClasswiseAugmentor(AugmentorBase):
    """Augments selected classes with a fixed number of augmentations per image"""

    def __init__(self, data_dir: str, n_augment: int, class_list: List[str], augmentation_type: str = "both"):
        super().__init__(data_dir, augmentation_type)
        self.n_augment = n_augment
        self.class_list = class_list

    def run(self):
        logging.info("Running classwise data augmentation...")
        for cls_name in self.class_list:
            cls_path = os.path.join(self.data_dir, cls_name)
            images = glob(os.path.join(cls_path, '*.png'))

            with tqdm(total=len(images), desc=f"Augmenting {cls_name}") as pbar:
                for idx, img_path in enumerate(images):
                    augmented = self._augment_image(img_path, self.n_augment, self.augmentation_pipeline)
                    self._save_augmented_images(augmented, cls_path, idx)
                    pbar.update()
        logging.info("Classwise augmentation complete.")
