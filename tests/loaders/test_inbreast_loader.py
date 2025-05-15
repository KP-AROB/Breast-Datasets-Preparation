import pandas as pd
import pytest
from src.core.df_loaders import INBreastDataframeLoader

# ---------------------------------------------------------------------
# 1. Successful `.load()` with valid .xls file
# ---------------------------------------------------------------------
def test_inbreast_loader_valid(tmp_path):
    path = tmp_path / "INbreast.xls"
    df = pd.DataFrame({
        'Bi-rads': [3, 2, 4],
        'Lesion annotation status': [1, None, 2]
    })
    df.to_excel(path, index=False)

    loader = INBreastDataframeLoader(str(tmp_path))
    result = loader.load()

    assert 'train' in result
    assert not result['train'].empty

# ---------------------------------------------------------------------
# 2. `.load()` raises exception if required XLS file is missing
# ---------------------------------------------------------------------
def test_inbreast_loader_missing_file(tmp_path):
    loader = INBreastDataframeLoader(str(tmp_path))
    with pytest.raises(Exception):
        loader.load()

# ---------------------------------------------------------------------
# 2. `.load()` raises exception if required columns are not present in the .xls file
# ---------------------------------------------------------------------
def test_inbreast_loader_missing_columns(tmp_path):
    path = tmp_path / "INbreast.xls"
    df = pd.DataFrame({
        'Wrong': [1, 2, 3]
    })
    df.to_excel(path, index=False)

    loader = INBreastDataframeLoader(str(tmp_path))
    with pytest.raises(Exception):
        loader.load()