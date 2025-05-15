import os
import ast
import pandas as pd
import logging


class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass


class FileMissingError(DatasetError):
    """Raised when a required file is missing."""
    pass


class ColumnMissingError(DatasetError):
    """Raised when required columns are missing."""
    pass


class EmptyDataFrameError(DatasetError):
    """Raised when a DataFrame is unexpectedly empty."""
    pass


def check_file_exists(path: str):
    if not os.path.exists(path):
        raise FileMissingError(f"File not found: {path}")


def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception as e:
        logging.warning(f"Failed to evaluate {val}: {e}")
        return []


def check_required_columns(df: pd.DataFrame, required_columns: set):
    missing = required_columns - set(df.columns)
    if missing:
        raise ColumnMissingError(f"Missing required columns: {missing}")


def check_non_empty_df(df: pd.DataFrame, name: str = "DataFrame"):
    if df.empty:
        raise EmptyDataFrameError(f"{name} is empty.")