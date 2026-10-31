# Social Media Shares Prediction

## Project Description
This project implements a machine learning pipeline to predict the number of shares (popularity) on social media platforms for the news articles.
The pipeline includes data cleaning, preprocessing, model training, and evaluation. Multiple regression models are trained and compared to find the best performing one based on various metrics.

## Prerequisites and Installation Instructions

### Requirements
- Anaconda or Miniconda
- Python 3.11
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:
    ```
    git clone https://github.com/jjwongjun/social-media-shares-prediction.git
    cd social-media-shares-prediction
    ```

2. Create and activate a conda environment:
    ```
    conda create -n sharesenv python=3.11
    conda activate sharesenv
    ```

3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Pipeline Execution
To run the complete pipeline, execute:
```
python main.py
```
A different configuration file can be used with `python main.py --config path/to/config.yaml` (default: `./src/config.yaml`). The pipeline expects the data at `./data/mini_project_1_data.csv` (see `file_path` in the config). The random seed, library versions and data shapes are logged at the start of each run.

Note: the hyperparameter grids are large (especially Gradient Boosting), so a full run can take a long time on a single CPU.

This will:
1. Validate the configuration and load the data
2. Validate and clean the data
3. Split data into training, validation, and test sets
4. Train baseline models (including a naive mean predictor as the minimum benchmark)
5. Perform hyperparameter tuning and report cross-validation results
6. Compare all models on the validation set and select the best one
7. Retrain the best model on training + validation data and evaluate it on the test set
8. Report feature importance of the final model

## Logical Flow of the Pipeline

### 1. Configuration (config.yaml)
The pipeline starts with loading configuration
parameters from `src/config.yaml` and checks that all required keys exist and have valid types and values. The config includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal)
- Train/validation/test split ratios
- Random seed (`seed`), used for the data splits and all models
- Hyperparameter grids for model tuning (`null` is used for "no limit" values, because `None` would be read as a text string in YAML)

### 2. Data Preparation (`DataPreparation` class)
The `DataPreparation` class handles:
- Validating the raw data (not empty, required columns present, target has no missing, infinite or negative values) and logging its shape and missing values per column
- Removing duplicates
- Setting negative `kw_min_min`, `kw_avg_min` and `kw_min_avg` values to 0
- Fill in 0 for null values of `num_imgs` and `num_videos`
- Removing rows with missing data (provided low-risk)
- Removing irrelevant features
- Creating a preprocessor for feature transformation (including imputation of missing values)

### 3. Model Training (`ModelTraining` class)
The `ModelTraining` class manages:
- Splitting data into training, validation, and test sets (the target is log-transformed)
- Training baseline models (Dummy mean predictor, Ridge, Random Forest, Gradient Boosting)
- Hyperparameter tuning for Ridge, Random Forest, Gradient Boosting models
- Reporting the mean and standard deviation of R², MAE and RMSE across the cross-validation folds for every model
- Evaluating models using multiple metrics (MAE, MSE, RMSE, R²)
- Selecting the best model based on validation R² score
- Retraining the selected model on training + validation data before the final test evaluation
- Reporting test errors in the original unit (shares) as well as on the log scale
- Reporting permutation feature importance of the final model

### 4. Main Execution (`main.py`)
The main script orchestrates the entire pipeline by:
- Loading and validating the configuration, and loading the data (with error handling)
- Initializing data preparation and model training
- Running the training and evaluation process
- Logging a side-by-side comparison of all models
- Identifying the best model, retraining it on training + validation data and evaluating it on the test set

## Key Findings and Feature Handling

### Exploratory Data Analysis

Before diving into the model training process, an exploratory data analysis (EDA) was conducted to understand the data better and identify any potential issues. Here are some key findings from the EDA:

**EDA Findings and Explanations**: 
- Many `kw_min_min` have negative values of only -1
- All negative `kw_avg_min` has also negative `kw_min_min` values
- Many `kw_avg_min` have negative values ranging from -1 to -0.051, suggesting that incorrect negative values of kw_min_min leaked into kw_avg_min
- For `kw_min_min` and `kw_avg_min` negative values if modified to 0, some kw_max_min are > 0. However, it is impossible to determine the real kw_avg_min values from dataset. Hence assume kw_avg_min to be 0 for simplicity, risk low with 767 over 30k+ rows
- Minimal amount of negative `kw_min_avg` values, to assume 0
- Check bounds `kw_min_min` <= `kw_avg_min` <= `kw_max_min`
- Replace null values of num_imgs and num_videos to 0 because plausible for article to not have images or video
- Null values of `self_reference_min_shares`, `self_reference_max_shares`, `self_reference_avg_shares`appear on the same row
- Cannot tell true value of `self_reference_min_shares`, `self_reference_max_shares`, `self_reference_avg_shares` from dataset, too coincidental for all 3 to be null at same time, to drop all 721 rows (2% of dataset, low risk)
- `data_channel` has null values
- Since URL of null data_channel suggest randomness, while box plot and bar chart suggests unwise to impute via mode (world), to drop data_channel with null values
- There are significantly more shares on weekdays vs weekends
- Distribution of `shares` is heavily right skewed. However, high shares values are plausible values, do not manipulate 
- Outlier `share` information do not show implausible data 
- No duplicates
- Moderate positive Pearson correlation between `shares` and `n_comments`
- Moderate positive Spearman Rank monotonic correlation between the `shares` and `n_comments`

### Data Cleaning
- Negative kw_min_avg, kw_min_min and kw_avg_min values set to 0 (clipped)
- num_imgs and num_videos null values replaced to 0
- Duplicate rows removed (none were found)
- Null values for self_reference_min_shares rows dropped
- Null value for data_channel rows dropped
- Irrevalant features ID and URL dropped 

### Feature Processing
- **Numerical Features**:
  - timedelta
  - n_tokens_title
  - n_tokens_content
  - n_unique_tokens
  - n_non_stop_words
  - n_non_stop_unique_tokens
  - num_hrefs
  - num_self_hrefs
  - num_imgs
  - num_videos
  - n_comments
  - average_token_length
  - self_reference_min_shares
  - self_reference_max_shares
  - self_reference_avg_shares
  - num_keywords
  - kw_min_min
  - kw_max_min
  - kw_avg_min
  - kw_min_max
  - kw_max_max
  - kw_avg_max
  - kw_min_avg
  - kw_max_avg
  - kw_avg_avg
- Missing values are imputed with the median (a safeguard, as the cleaning step already removes rows with missing values). The features are then standardized using `StandardScaler` because scaling matters for Ridge which penalizes coefficient size, so it needs features on comparable scales. Random Forest and gradient boosting are unaffected by scaling, so a shared preprocessor with StandardScaler does no harm to them.

- **Nominal Features**: `data_channel`, `weekday` - Missing values are imputed with the most frequent category, then encoded using `OneHotEncoder` because these features have no intrinsic order
- **Ordinal Features**: No ordinal features
- **Passthrough Features**: No passthrough features
- The preprocessor is part of every model pipeline, so it is fitted on the training data only (no data leakage). Its output is dense so that tree-based models can use it

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Dummy (mean) predictor**: Always predicts the mean of the training target. It is the minimum benchmark, so every real model must clearly beat it (R² of about 0) to be useful
2. **Ridge Regression**: Linear regression with L2 regularization because it includes a regularization term to prevent overfitting. Since different input features will have varying levels of influence on the resale price, the reduction of coefficient size caused by the shrinkage penalty translates to each predictor feature having less influence on the final prediction
3. **Random Forest**: Non-linear model that averages many trees. It can capture effects that aren’t straight lines and interactions between features. It’s also fairly robust to outliers and needs little tuning. Averaging many deep trees reduces overfitting, so it’s a safe non-linear option.
4. **Gradient boosting**: a non-linear model that builds trees one after another.
Each new tree corrects the errors of the previous ones.

### Hyperparameter Tuning
- Grid search is performed for all models because tuning model parameters can achieve the best possible performance. Grid Search CV's process is exhaustive but it ensures the best combination
- Parameters tuned cover a wide range because the right combination of these settings ensures balance and should perform well on unseen data
- 5-fold cross-validation is used during tuning because averaged across all 5 folds will give a more reliable estimate of model performance. The mean and standard deviation of R², MAE and RMSE across the folds are logged for every model, baselines included
- The best model is selected on the validation set and then retrained on training + validation data before it is evaluated once on the test set

### Evaluation Metrics
- Because `shares` is log-transformed (`log1p`) before modelling, the metrics are on the log scale. For the final model, MAE and RMSE are also reported in the original unit (shares)
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same unit as the target
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model

## Deployment Considerations
- Code is structured into modular well-organized components making code easier to maintain, more scalable, and ready for real world applications
- The configuration and data are validated, and file loading and pipeline errors are logged with clear messages
- Possible next steps: unit tests for the cleaning functions, saving the fitted model and metrics, and experiment tracking across runs