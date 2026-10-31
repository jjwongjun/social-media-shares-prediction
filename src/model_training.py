# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline


class ModelTraining:
    """
    A class used to train and evaluate machine learning models on the social media shares data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training and evaluation.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical and nominal features.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initialize the ModelTraining class with configuration and preprocessor.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        preprocessor (sklearn.compose.ColumnTransformer): A preprocessor pipeline for transforming numerical and nominal features.
        """
        self.config = config
        self.preprocessor = preprocessor

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split the data into training, validation, and test sets.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the cleaned data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: A tuple containing the training, validation, and test features and target variables.
        """
        logging.info("Starting data splitting.")
        if self.config["target_column"] not in df.columns:
            raise ValueError(f"Target column '{self.config['target_column']}' not found.")
        X = df.drop(columns=self.config["target_column"])
        # The target is right-skewed, so it is log-transformed (metrics are on the log scale)
        y = np.log1p(df[self.config["target_column"]])
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config["val_test_size"], random_state=self.config["seed"]
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config["val_size"], random_state=self.config["seed"]
        )
        logging.info("Data split into train, validation, test sets.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Create, train, and evaluate baseline models.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline models.")
        seed = self.config["seed"]
        models = {
            # Naive benchmark: always predicts the mean of the training target
            "dummy_mean": DummyRegressor(strategy="mean"),
            "ridge": Ridge(alpha=10),
            "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                           n_jobs=1, random_state=seed),
            "gradient_boost": HistGradientBoostingRegressor(random_state=seed),
        }
        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("regressor", model)]
            )
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline
            metrics[model_name] = self._evaluate_model(
                pipeline, X_val, y_val, model_name
            )
            self._log_cv_summary(pipeline, X_train, y_train, model_name)

        return pipelines, metrics

    def train_and_evaluate_tuned_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Perform hyperparameter tuning for Ridge, Random Forest and Gradient Boosting models and evaluate them.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the tuned pipelines and their evaluation metrics.
        """
        logging.info("Starting hyperparameter tuning.")
        tuned_models = {}
        tuned_metrics = {}
        ridge_param_grid = self.config["ridge_param_grid"]
        rf_param_grid = self.config["rf_param_grid"]
        gb_param_grid = self.config["gb_param_grid"]
        param_grids = {
            "ridge_tuned": ridge_param_grid,
            "rf_tuned": rf_param_grid,
            "gb_tuned": gb_param_grid,
        }
        cv = self.config["cv"]
        scoring = self.config["scoring"]
        seed = self.config["seed"]

        models = {"ridge_tuned": Ridge(), 
                  "rf_tuned": RandomForestRegressor(n_estimators=300, n_jobs=1, random_state=seed), 
                  "gb_tuned": HistGradientBoostingRegressor(random_state=seed)}

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("regressor", model)]
            )
            grid_search = GridSearchCV(
                pipeline, param_grids[model_name], cv=cv, scoring=scoring, n_jobs=1
            )
            grid_search.fit(X_train, y_train)
            logging.info(f"{model_name} best parameters: {grid_search.best_params_}")
            tuned_models[model_name] = grid_search.best_estimator_
            self._log_cv_summary(tuned_models[model_name], X_train, y_train, model_name)
            tuned_metrics[model_name] = self._evaluate_model(
                tuned_models[model_name], X_val, y_val, model_name + " (tuned)"
            )

        logging.info("Hyperparameter tuning completed.")
        return tuned_models, tuned_metrics

    def evaluate_final_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate the final model on the test set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        y_test_pred = model.predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "MSE": mean_squared_error(y_test, y_test_pred),
            "RMSE": root_mean_squared_error(y_test, y_test_pred),
            "R²": r2_score(y_test, y_test_pred),
        }
        # The target was log-transformed, so also report errors in the original unit (shares)
        y_test_shares = np.expm1(y_test)
        y_pred_shares = np.expm1(y_test_pred)
        metrics["MAE (shares)"] = mean_absolute_error(y_test_shares, y_pred_shares)
        metrics["RMSE (shares)"] = root_mean_squared_error(y_test_shares, y_pred_shares)
        logging.info(f"Final Test Metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics

    def retrain_on_train_val(
        self,
        model: Pipeline,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> Pipeline:
        """
        Retrain the selected model (same hyperparameters) on train + validation data.

        Args:
        -----
        model (Pipeline): The selected model pipeline.
        X_train, X_val (pd.DataFrame): The training and validation features.
        y_train, y_val (pd.Series): The training and validation target variables.

        Returns:
        --------
        Pipeline: A new pipeline fitted on the combined data.
        """
        logging.info("Retraining best model on training + validation data.")
        final_model = clone(model)
        final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        return final_model

    def log_feature_importance(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.Series:
        """
        Log permutation importance (drop in R² when a feature is shuffled) on the
        test set. Works for any model.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.

        Returns:
        --------
        pd.Series: Feature importances sorted from most to least important.
        """
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            scoring="r2",
            random_state=self.config["seed"],
            n_jobs=1,
        )
        importances = pd.Series(result.importances_mean, index=X_test.columns)
        importances = importances.sort_values(ascending=False)
        logging.info(f"Permutation feature importance (top 15):\n{importances.head(15).round(4)}")
        return importances

    def _log_cv_summary(
        self, model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, model_name: str
    ) -> None:
        """
        Log the mean and standard deviation of R², MAE and RMSE across CV folds.

        Args:
        -----
        model (Pipeline): The model pipeline.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        model_name (str): The name of the model being evaluated.
        """
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=self.config["cv"],
            scoring={
                "R²": "r2",
                "MAE": "neg_mean_absolute_error",
                "RMSE": "neg_root_mean_squared_error",
            },
            n_jobs=1,
        )
        logging.info(f"{model_name} {self.config['cv']}-fold CV results:")
        for metric in ["R²", "MAE", "RMSE"]:
            values = scores[f"test_{metric}"]
            if metric != "R²":
                values = -values
            logging.info(f"{metric}: {values.mean():.4f} (+/- {values.std():.4f})")

    def _evaluate_model(
        self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a model on the validation set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        y_val_pred = model.predict(X_val)
        metrics = {
            "MAE": mean_absolute_error(y_val, y_val_pred),
            "MSE": mean_squared_error(y_val, y_val_pred),
            "RMSE": root_mean_squared_error(y_val, y_val_pred),
            "R²": r2_score(y_val, y_val_pred),
        }
        logging.info(f"{model_name} Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics