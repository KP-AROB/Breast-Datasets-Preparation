import logging, json, h5py, math, os, shutil, gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from .base import BaseConverter
from src.processing.pipeline import BasePipeline
from src.utils.io import preload_to_local
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed

class VindrH5Converter(BaseConverter):
    def __init__(self, processing_pipeline: BasePipeline, batch_size: int = 500, n_workers: int = 4, tmp_dir: str = None):
        super().__init__(processing_pipeline, batch_size, n_workers, tmp_dir)
        
        self.birads_mapping = {
            'bi-rads_1': '1',
            'bi-rads_2': '2',
            'bi-rads_3': '0',
            'bi-rads_4': '0',
            'bi-rads_5': '0'
        }
    
        self.lesions_mapping = {
            'no_finding': '0',
            'mass': '1',
            'suspicious_calcifications': '2'
        }
        
    def write(self, 
            filename: str, 
            image_batch: List[np.ndarray],
            birads_batch: List[int], 
            lesions_batch: List[int],
        ) -> None:
        if not image_batch:
            logging.warning(f"No images to write to {filename}")
            return
        with h5py.File(filename, 'w') as h5_file:
            h5_file.create_dataset("x", data=np.array(image_batch), compression="gzip")
            birads_dataset = h5_file.create_dataset("y_birads", data=np.array(birads_batch, dtype=np.int32), compression="gzip")
            lesions_dataset = h5_file.create_dataset("y_lesions", data=np.array(lesions_batch, dtype=np.int32), compression="gzip")
            birads_dataset.attrs['label_mapping'] = json.dumps(self.birads_mapping)
            lesions_dataset.attrs['label_mapping'] = json.dumps(self.lesions_mapping)
    
    def _process_batch(self, file_paths: List[str], output_dir: str, birads: List[int], lesions: List[int]) -> None:
        num_chunks = math.ceil(len(file_paths) / self.batch_size)
        with tqdm(total=num_chunks, desc="Processing batches") as pbar:
            for idx in range(num_chunks):
                batch_paths = list(file_paths[idx * self.batch_size:(idx + 1) * self.batch_size])
                batch_lesions = list(lesions[idx * self.batch_size:(idx + 1) * self.batch_size])
                batch_birads = list(birads[idx * self.batch_size:(idx + 1) * self.batch_size])

                description_prefix = f"Chunk {idx}/{num_chunks}"
                batch_images = []
                valid_lesions = []
                valid_birads = []

                try:
                    if self.tmp_dir:
                        pbar.set_description(f"{description_prefix} - Copying to temp dir")
                        batch_paths, _ = preload_to_local(batch_paths, custom_dir=self.tmp_dir, max_files=self.batch_size)

                    pbar.set_description(f"{description_prefix} - Processing")
                    t_start = perf_counter()
                    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                        futures = {
                            executor.submit(self.processing_pipeline.process, path): i
                            for i, path in enumerate(batch_paths)
                        }

                        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batch", leave=False):
                            i = futures[future]
                            path = batch_paths[i]
                            try:
                                image = future.result()
                                if image is not None:
                                    batch_images.append(image)
                                    valid_birads.append(batch_birads[i])
                                    valid_lesions.append(batch_lesions[i])
                            except RuntimeError as e:
                                logging.warning(f"Skipping {path} due to error: {e}")
                            except Exception as e:
                                logging.error(f"Unexpected error on {path}: {e}")

                    if not batch_images:
                        logging.error("No images were processed successfully in this batch.")
                        return

                    pbar.set_description(f"{description_prefix} - Saving batch to HDF5")
                    filename = os.path.join(output_dir, f"batch_{idx:04d}.h5")
                    self.write(filename, batch_images, valid_birads, valid_lesions)

                finally:
                    if self.tmp_dir:
                        pbar.set_description(f"{description_prefix} - Cleaning up temp dir")
                        shutil.rmtree(self.tmp_dir)
                    del batch_images
                    gc.collect()

                    pbar.set_description(f"{description_prefix} - Done in {perf_counter() - t_start:.2f}s")
                    pbar.update()
    
    def run(self, dataframes: Dict[str, pd.DataFrame], output_dir: str) -> None:
        for df_name, df in dataframes.items():
            logging.info(f"Processing {df_name} dataframe")
            row_indices = df.index.tolist()
            print(df.keys())
            paths = df['absolute_path'][row_indices]
            birads = df['breast_birads'][row_indices]
            lesions = df['finding_categories'][row_indices]
            save_dir = os.path.join(output_dir, df_name)
            self._init(paths, save_dir)
            logging.info(f'Saving files from {df_name} dataframe')
            self._process_batch(paths, save_dir, birads, lesions)
            logging.info(f"All batches from dataframe '{df_name}' processed successfully.")