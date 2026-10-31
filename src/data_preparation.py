# Standard library imports
import logging
from typing import Any, Dict

# Related third-party imports
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreparation:
    """
    A class used to clean and preprocess the social media shares data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical and nominal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        self._validate_data(df)
        df = df.copy()
        rows_before = len(df)
        df = df.drop_duplicates()
        logging.info(f"Removed {rows_before - len(df)} duplicate rows.")
        # Negative keyword values are invalid, so they are set to 0
        kw_columns = ["kw_min_min", "kw_avg_min", "kw_min_avg"]
        df[kw_columns] = df[kw_columns].clip(lower=0)
        df[["num_imgs", "num_videos"]] = df[["num_imgs", "num_videos"]].fillna(0)
        df = df.dropna(subset=["self_reference_min_shares"]).reset_index(drop=True)
        df = df.dropna(subset=["data_channel"]).reset_index(drop=True)
        df.drop(columns=["ID", "URL"], inplace=True)
        logging.info(f"Cleaned data shape: {df.shape}")
        logging.info("Data cleaning completed.")
        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validates the raw data before cleaning: checks that it is not empty, that all
        required columns exist and that the target has no missing or infinite values.
        Also logs the shape and the missing values per column.

        Args:
        -----
        df (pd.DataFrame): The raw DataFrame to validate.

        Raises:
        -------
        ValueError: If the data is empty, columns are missing or the target is invalid.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        target = self.config["target_column"]
        required_columns = (
            {target, "ID", "URL", "num_imgs", "num_videos", "self_reference_min_shares"}
            | {"kw_min_min", "kw_avg_min", "kw_min_avg"}
            | set(self.config["numerical_features"])
            | set(self.config["nominal_features"])
        )
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Input data is missing columns: {sorted(missing_columns)}")
        if df[target].isna().any() or not np.isfinite(df[target]).all():
            raise ValueError(f"Target column '{target}' has missing or infinite values.")
        if (df[target] < 0).any():
            raise ValueError(f"Target column '{target}' has negative values.")
        logging.info(f"Raw data shape: {df.shape}")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        logging.info(f"Missing values per column: {missing.to_dict()}")

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical and nominal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        nominal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"])
            ],
            sparse_threshold=0,  # dense output so tree-based models can be used
            n_jobs=1,
        )
        return preprocessor