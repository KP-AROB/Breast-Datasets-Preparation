import pandas as pd
from typing import Callable, Dict, Type
from .base import BaseDataframeLoader

_LOADER_REGISTRY: Dict[str, Callable[[str], BaseDataframeLoader]] = {}

def register_loader(name: str):
    def decorator(cls: Type[BaseDataframeLoader]):
        _LOADER_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def get_dataset_loader(name: str, data_dir: str) -> BaseDataframeLoader:
    try:
        loader_class = _LOADER_REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown dataset name: {name}. Available: {list(_LOADER_REGISTRY)}")
    return loader_class(data_dir)

def get_datasets(name: str, data_dir: str) -> Dict[str, pd.DataFrame]:
    loader = get_dataset_loader(name, data_dir)
    return loader.load()