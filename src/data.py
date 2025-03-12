import pandas as pd


def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)


def clean_data(df):
    """Handle missing values and drop unnecessary columns."""
    return df
