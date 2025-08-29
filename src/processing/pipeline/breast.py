from .base import BasePipeline
from src.processing.operations.read import read_dicom
from src.processing.operations.transform import crop_to_roi, resize, build_pyramid
from src.processing.operations.normalize import truncate_normalization, clahe, minmax_normalisation, apply_gabor

class BreastImageProcessingPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.operations = [
            read_dicom,
            minmax_normalisation, 
            #apply_gabor,
            crop_to_roi,
            resize,
            #clahe,
        ]
