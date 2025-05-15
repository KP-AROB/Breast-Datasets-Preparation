import pandas as pd
import os
from .base import BaseDataframeLoader
from sklearn.model_selection import train_test_split
from typing import Dict
from src.utils.errors import *

class INBreastDataframeLoader(BaseDataframeLoader):
    """
    INBreast DataFrame loader.
    """
    def __init__(self, data_dir: str):
        """
        Initialize the INBreast DataFrame loader
        """
        super().__init__(data_dir)
 
    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Load the INBreast DataFrame.
        """
        filepath = os.path.join(self.data_dir, 'INbreast.xls')
        check_file_exists(filepath)
        df = pd.read_excel(filepath, skipfooter=2)
        df.columns = df.columns.str.strip().str.capitalize()
        
        df = df[df["Bi-rads"].notna()]
        df["Lesion annotation status"] = df["Lesion annotation status"].fillna(1)
        df.loc[df["Lesion annotation status"] != 1, "Lesion annotation status"] = 0
        train, test = train_test_split(df, test_size=0.2) if len(df) > 1 else (df, df)
        check_non_empty_df(train, 'Training Dataframe')
        check_non_empty_df(test, 'Training Dataframe')
        return {'train': train, 'val': test, 'test': test}
