import pytest
from src.core.df_loaders import VindrDataframeLoader



@pytest.fixture
def fake_good_finding_annotations(tmp_path):
    path = tmp_path / "finding_annotations.csv"
    path.write_text("""study_id,image_id,split,finding_categories,breast_birads
    s1,img1,training,"['mass']",bi-rads_2
    s2,img2,test,"['mass']",bi-r"ads_2
    """)
    return tmp_path

# ---------------------------------------------------------------------
# 1. Successful `.load()` with valid input CSV containing expected categories
# ---------------------------------------------------------------------
def test_vindr_loader_valid(fake_good_finding_annotations):
    loader = VindrDataframeLoader(str(fake_good_finding_annotations))
    result = loader.load()

    assert 'train' in result and not result['train'].empty
    assert 'test' in result and not result['test'].empty

# ---------------------------------------------------------------------
# 2. `.load()` raises exception if required CSV file is missing
# ---------------------------------------------------------------------
def test_vindr_loader_missing_file(tmp_path):
    loader = VindrDataframeLoader(str(tmp_path))
    with pytest.raises(Exception):
        loader.load()