from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from scipy.stats import boxcox

# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine


def detect_outliers_iqr(df: pd.DataFrame, threshold: float = 1.5) -> dict:
    """
    Detect outliers in numerical columns using the IQR method.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Multiplier for the IQR to define outliers (default is 1.5).
    
    Returns:
        dict: Dictionary with column names as keys and count of outliers as values.
    """
    outlier_counts = {}
    numeric_cols = df.select_dtypes(include=[np.number])

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = outliers.shape[0]
    
    return {col: count for col, count in outlier_counts.items() if count > 0}


def apply_transformations(df: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Apply transformations to reduce outliers' impact on numerical columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        method (str): Transformation method ('log', 'sqrt', 'boxcox').

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    df_transformed = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number])

    for col in numeric_cols:
        if method == "log":
            df_transformed[col] = np.log1p(df[col])
        elif method == "sqrt":
            df_transformed[col] = np.sqrt(df[col])
        elif method == "boxcox":
            df_transformed[col], _ = boxcox(df[col] + 1)
        else:
            raise ValueError("Invalid method. Choose 'log', 'sqrt', or 'boxcox'.")

    return df_transformed
