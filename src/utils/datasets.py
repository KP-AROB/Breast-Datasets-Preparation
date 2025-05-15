import pandas as pd
from typing import Dict
from src.core.df_loaders import VindrDataframeLoader, INBreastDataframeLoader

def get_datasets(name: str, data_dir: str) -> Dict[str, pd.DataFrame]:
    if name == "vindr":
        loader = VindrDataframeLoader(data_dir)
        dataframes = loader.load()
    elif name == "inbreast":
        loader = INBreastDataframeLoader(data_dir)
        dataframes = loader.load()
    return dataframes
