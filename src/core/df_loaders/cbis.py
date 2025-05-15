import os
import logging
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import Dict
from .base import BaseDataframeLoader

class CBISDataframeLoader(BaseDataframeLoader):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)

        if len(glob(os.path.join(data_dir, '*corrected.csv'))) != 4:
            logging.info('Corrected csv files not found. Creating ...')
            self.correct_metadata_files()
            logging.info('Corrected csv files created.')

    def load(self) -> Dict[str, pd.DataFrame]:
        def load_and_process(*filenames):
            dfs = [pd.read_csv(os.path.join(self.data_dir, f)) for f in filenames]
            df = pd.concat(dfs, ignore_index=True)
            return self.make_cls_column(df)

        train_files = [
            'mass_case_description_train_set_corrected.csv',
            'calc_case_description_train_set_corrected.csv'
        ]
        test_files = [
            'mass_case_description_test_set_corrected.csv',
            'calc_case_description_test_set_corrected.csv'
        ]

        train_df = load_and_process(*train_files)
        test_df = load_and_process(*test_files)

        return {
            'train': train_df,
            'val': test_df,
            'test': test_df
        }

    def make_cls_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df['pathology'] = df['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
        df['pathology'] = df['abnormality type'] + '_' + df['pathology']
        return df

    def normalize_and_format_path(self, path: str) -> str:
        if path.startswith(".\\"):
            path = path[2:]
        path = path.replace("\\", "/")
        path_parts = path.split("/")
        if path_parts:
            last_part = path_parts[-1]
            number, *rest = last_part.split("-", 1)
            if number.isdigit():
                number = number.zfill(2)
            path_parts[-1] = f"{number}-{''.join(rest)}"
        return "/".join(path_parts)

    def get_image_path_ids(self, row, key):
        path = row[key]
        path_segment = path.split(os.sep)
        study_id = path_segment[1]
        series_uid = path_segment[2]
        return study_id, series_uid

    def correct_metadata_files(self):
        metadata_df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))

        lesion_files = {
            f"{desc}_case_description_{set_type}_set":
                os.path.join(self.data_dir, f"{desc}_case_description_{set_type}_set.csv")
            for desc in ["mass", "calc"]
            for set_type in ["train", "test"]
        }

        with tqdm(total=len(lesion_files), desc='Correcting CBIS csv files') as pbar:
            for key, path in lesion_files.items():
                df = pd.read_csv(path)
                df = df.rename(columns={
                    'left or right breast': 'left_or_right_breast',
                    'image view': 'image_view',
                    'abnormality id': 'abnormality_id',
                    'mass shape': 'mass_shape',
                    'mass margins': 'mass_margins',
                    'image file path': 'image_file_path',
                    'cropped image file path': 'cropped_image_file_path',
                    'ROI mask file path': 'roi_mask_file_path'
                })

                for idx, row in df.iterrows():
                    for field in ['image_file_path', 'roi_mask_file_path', 'cropped_image_file_path']:
                        study_id, series_uid = self.get_image_path_ids(row, field)
                        meta = metadata_df[(metadata_df['Series UID'] == series_uid) & (metadata_df['Study UID'] == study_id)]
                        if not meta.empty:
                            df.at[idx, field] = self.normalize_and_format_path(meta['File Location'].values[0])

                corrected_path = os.path.join(self.data_dir, f"{key}_corrected.csv")
                df.to_csv(corrected_path, index=False)
                pbar.update()