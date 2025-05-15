from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class BaseDataframeLoader(ABC):
    def __init__(self, data_dir: str, is_train: bool = True):
        self.data_dir = data_dir
        self.is_train = is_train

    @abstractmethod
    def load(self) -> Dict[str, pd.DataFrame]:
        """Return a dictionary with keys like 'train', 'val', and 'test'."""
        pass