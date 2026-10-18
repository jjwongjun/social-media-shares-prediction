# Standard library imports
import logging
import re
from typing import Any, Dict

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class DataPreparation:
    """
    A class used to clean and preprocess HDB resale prices data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
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
        df[["kw_min_min", "kw_avg_min", "kw_min_avg"]].clip(lower=0)
        df[["num_imgs", "num_videos"]] = df[["num_imgs", "num_videos"]].fillna(0)
        df = df.dropna(subset=["self_reference_min_shares"]).reset_index(drop=True)
        df = df.dropna(subset=["data_channel"]).reset_index(drop=True)
        df.drop(columns=["ID", "URL"], inplace=True)
        logging.info("Data cleaning completed.")
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"])
            ],
            n_jobs=1,
        )
        return preprocessor