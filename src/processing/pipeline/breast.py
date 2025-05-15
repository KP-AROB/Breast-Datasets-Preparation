from .base import BasePipeline
from src.processing.operations.read import read_dicom
from src.processing.operations.transform import crop_to_roi, resize
from src.processing.operations.normalize import truncate_normalization

class BreastImageProcessingPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.operations = [
            read_dicom,
            crop_to_roi,
            truncate_normalization,
            resize,
        ]
