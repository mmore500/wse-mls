import pandas as pd

import pylib


def test_osf_fetch():
    cache_path = pylib.osf_fetch("pvq7w")
    assert cache_path.exists()
    assert cache_path.stat().st_size > 0

    df = pd.read_csv(cache_path)
    assert len(df) > 0
