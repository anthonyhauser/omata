# data_utils/load.py

import pandas as pd

def load_csv(path):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)