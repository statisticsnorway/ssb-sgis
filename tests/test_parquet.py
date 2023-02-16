from pathlib import Path

import pandas as pd


def test_read_parquet_test_file() -> None:
    filename = Path(__file__).parent / "testdata" / "example.parquet"
    df = pd.read_parquet(filename)
    assert df.shape[0] == 3  # Three rows
