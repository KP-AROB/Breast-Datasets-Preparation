import logging, json, h5py
import numpy as np
import pandas as pd

from typing import List, Dict
from .base import BaseConverter
from src.processing.pipeline import BasePipeline

class VindrH5Converter(BaseConverter):
    def __init__(self, processing_pipeline: BasePipeline, batch_size: int = 500, n_workers: int = 4, tmp_dir: str = None):
        super().__init__(processing_pipeline, batch_size, n_workers, tmp_dir)
    
    def write(self, 
            filename: str, 
            image_batch: List[np.ndarray],
            birads_batch: List[int], 
            lesions_batch: List[int],
            birads_mapping: dict = None,
            lesions_mapping: dict = None,
        ) -> None:
        if not image_batch:
            logging.warning(f"No images to write to {filename}")
            return
        with h5py.File(filename, 'w') as h5_file:
            h5_file.create_dataset("x", data=np.array(image_batch), compression="gzip")
            birads_dataset = h5_file.create_dataset("y_birads", data=np.array(birads_batch, dtype=np.int32), compression="gzip")
            lesions_dataset = h5_file.create_dataset("y_lesions", data=np.array(lesions_batch, dtype=np.int32), compression="gzip")
            birads_dataset.attrs['label_mapping'] = json.dumps(birads_mapping)
            lesions_dataset.attrs['label_mapping'] = json.dumps(lesions_mapping)
    
    def run(self, dataframe: Dict[str, pd.DataFrame], output_dir: str) -> None:
        for df_name, df in dataframe.items():
            logging.info(f"Processing {df_name} dataframe")
            paths = df.index.tolist()
            birads_dict = df['breast_birads'].to_dict()
            lesions_dict = df['finding_categories'].to_dict()
            self._init(paths, output_dir)
            self._process_batch(paths, output_dir, birads_dict, lesions_dict)
            logging.info(f"All batches from dataframe '{df_name}' processed successfully.")
