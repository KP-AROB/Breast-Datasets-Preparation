import os
import logging
import concurrent.futures

from glob import glob
from tqdm import tqdm
from typing import List
from collections import Counter
from typing import Dict
from .base import AugmentorBase

class BalancedAugmentor(AugmentorBase):
    """Over-samples under-represented classes to balance the dataset"""

    def __init__(self, data_dir: str, dataset_targets: List[int], augmentation_type: str = "both"):
        super().__init__(data_dir, augmentation_type)
        self.class_counts: Dict[int, int] = Counter(dataset_targets)
        self.max_count = max(self.class_counts.values())

    def augment_class(self, class_label: int, n_augment: int):
        cls_path = os.path.join(self.data_dir, str(class_label))
        images = glob(os.path.join(cls_path, '*.png'))

        with tqdm(total=len(images), desc=f"Augmenting class {class_label} x{n_augment}") as pbar:
            for idx, img_path in enumerate(images):
                augmented = self._augment_image(img_path, n_augment, self.augmentation_pipeline)
                self._save_augmented_images(augmented, cls_path, idx)
                pbar.update()

    def run(self):
        logging.info("Running balanced data augmentation...")
        under_sampled = {cls: count for cls, count in self.class_counts.items() if count < self.max_count}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for cls, count in under_sampled.items():
                n_augment = int(self.max_count / count)
                futures.append(executor.submit(self.augment_class, cls, n_augment))
            for future in concurrent.futures.as_completed(futures):
                future.result()
        logging.info("Balanced augmentation complete.")