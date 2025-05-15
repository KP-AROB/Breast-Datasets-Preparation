import math, os, gc, shutil, logging
from typing import List, Any
from tqdm import tqdm
from abc import ABC, abstractmethod
from src.processing.pipeline import BasePipeline
from src.utils.io import preload_to_local
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

class BaseConverter(ABC):
    def __init__(self, processing_pipeline: BasePipeline, batch_size: int = 500, n_workers: int = 4, tmp_dir: str = None):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.tmp_dir = tmp_dir
        self.processing_pipeline = processing_pipeline
    
    @abstractmethod
    def run(self, output_dir: str):
        pass
        
    @abstractmethod
    def write(self, filename: str, batch_images: list, *args: Any, **kwargs: Any):
        pass
    
    def _init(self, paths: List[str], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        num_batches = math.ceil(len(paths) / self.batch_size)
        logging.info(f"Total files: {len(paths)}")
        logging.info(f"Processing in {num_batches} batches of {self.batch_size}")
        logging.info(f"Using {self.n_workers} workers")
        logging.info(f"Output directory: {output_dir}")
    
    def _process_batch(self, file_paths: List[str], output_dir: str) -> None:
        num_chunks = math.ceil(len(file_paths) / self.batch_size)
        with tqdm(total=num_chunks, desc="Processing batches") as pbar:
            for idx in range(num_chunks):
                batch_paths = file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
                description_prefix = f"Chunk {idx}/{num_chunks}"

                try:
                    if self.tmp_dir:
                        pbar.set_description(f"{description_prefix} - Copying to temp dir")
                        batch_paths, _ = preload_to_local(batch_paths, custom_dir=self.tmp_dir, max_files=self.batch_size)

                    pbar.set_description(f"{description_prefix} - Processing")
                    t_start = perf_counter()
                    with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                        batch_images = list(
                            tqdm(
                                executor.map(lambda path: self.processing_pipeline.process(path), batch_paths),
                                total=len(batch_paths),
                                leave=False,
                            )
                        )
                    
                    # TODO: Add ProcessPoolExecutor for heavy pipeline
                    
                    batch_images = [img for img in batch_images if img is not None]
                    pbar.set_description(f"{description_prefix} - Saving batch to HDF5")    
                    filename = os.path.join(output_dir, f"batch_{idx:04d}.h5")
                    self.write(filename, batch_images)
                finally:
                    if self.tmp_dir:
                        pbar.set_description(f"{description_prefix} - Cleaning up temp dir")
                        shutil.rmtree(self.tmp_dir)
                    del batch_images
                    gc.collect()

                    pbar.set_description(f"{description_prefix} - Done in {perf_counter() - t_start:.2f}s")
                    pbar.update()
                    
