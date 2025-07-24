import pandas as pd
import os, ast
from .base import BaseDataframeLoader
from typing import Dict
from src.utils.errors import *

class VindrDataframeLoader(BaseDataframeLoader):
    """
    Vindr-Mammo DataFrame loader.
    """
    def __init__(self, data_dir: str, birads_mapping: dict = None, lesions_mapping: dict = None):
        """
        Initialize the Vindr-Mammo DataFrame loader
        """
        super().__init__(data_dir)

        self.birads_mapping = {
            'bi-rads_1': '1',
            'bi-rads_2': '2',
            'bi-rads_3': '0',
            'bi-rads_4': '0',
            'bi-rads_5': '0'
        } if not birads_mapping else birads_mapping
    
        self.lesions_mapping = {
            'no_finding': '0',
            'mass': '1',
            'suspicious_calcification': '2'
        } if not lesions_mapping else lesions_mapping
    
    def format_char(self, char):
        return char.lower().replace(' ', '_')

    def format_category_list(self, category_list):
        return [self.format_char(category) for category in category_list]

    def contains_all_classes(self, category_list, class_list):
        return any(cls in category_list for cls in class_list)

    def replace_categories(self, df, column, target_categories):
        def replace_if_present(categories):
            for target in target_categories:
                if target in categories:
                    return target
            return categories

        df[column] = df[column].apply(
            lambda x: replace_if_present(x) if isinstance(x, list) else x)

    def _construct_image_path(self, row: pd.Series) -> str:
        path = os.path.join(self.data_dir, 'images', row['study_id'], row['image_id'] + '.dicom')
        return path
        
    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Load the Vindr-Mammo DataFrame.
        """
        filepath = os.path.join(self.data_dir, 'finding_annotations.csv')
        check_file_exists(filepath)
        df_find = pd.read_csv(filepath)
        check_required_columns(df_find, {'study_id', 'image_id', 'finding_categories', 'breast_birads', 'split'})

        df_find['finding_categories'] = df_find['finding_categories'].apply(ast.literal_eval)
        df_find['finding_categories'] = df_find['finding_categories'].apply(self.format_category_list)
        df_find['breast_birads'] = df_find['breast_birads'].apply(self.format_char)
        df_find['breast_birads'] = df_find['breast_birads'].replace(self.birads_mapping)
        df_find.drop_duplicates(subset='image_id', keep=False, inplace=True)

        # Replace and filter finding categories
        target_categories = ['mass', 'no_finding', 'suspicious_calcification']
        self.replace_categories(df_find, 'finding_categories', target_categories)
        df_find = df_find[df_find['finding_categories'].isin(target_categories)]
        df_find['finding_categories'] = df_find['finding_categories'].replace(self.lesions_mapping)

        return {'train': df_find, 'val': df_find}