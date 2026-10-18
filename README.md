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

git clone https://github.com/jjwongjun/social-media-shares-prediction.git
cd social-media-shares-prediction

2. Create and activate a conda environment:
    ```
    conda create -n hdbenv python=3.11
    conda activate hdbenv
    ```

3. Install the required packages:

pip install -r requirements.txt

## Pipeline Execution
To run the complete pipeline, execute:

python main.py

This will:
1. Load and clean the data
2. Preprocess features
3. Split data into training, validation, and test sets
4. Train baseline models
5. Perform hyperparameter tuning
6. Evaluate models and select the best one

## Logical Flow of the Pipeline

### 1. Configuration (config.yaml)
The pipeline starts with loading configuration
parameters from `src/config.yaml`, which includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, ordinal)
- Train/validation/test split ratios
- Hyperparameter grid for model tuning

### 2. Data Preparation (`DataPreparation` class)
The DataPreparation' class handles:
- Fill in 0 for null values
- Removing rows with missing data (provided low-risk)
- Converting storey ranges to numerical values
- Removing irrelevant features
- Removing duplicates
- Creating a preprocessor for feature transformation

### 3. Model Training (`ModelTraining` class)
The ModelTraining' class manages:
- Splitting data into training, validation, and test
sets
- Training baseline models (Ridge, Random Forest, Gradient Boosting)
- Hyperparameter tuning for Ridge, Random Forest, Gradient Boosting models
- Evaluating models using multiple metrics (MAE, MSE, RMSE, R²)
- Selecting the best model based on R² score

### 4. Main Execution (`main.py`)
The main script orchestrates the entire pipeline by:
- Loading configuration and data
- Initializing data preparation and model training
- Running the training and evaluation process
- Identifying and evaluating the best model on the test set

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
- kw_min_avg, kw_min_min, kw_avg_min, num_imgs, num_videos null values replaced to 0
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
- Standardized using `StandardScaler` because many numerical features have outliers and StandardScaler is less sensitive to outliers as opposed to normalization. approximately normal distributions. In addition, scaling only matters for Ridge which penalizes coefficient size, so it needs features on comparable scales. Random Forest and gradient boosting are unaffected by scaling, so a shared preprocessor with StandardScaler does no harm to them.

- **Nominal Features**: `data_channel`, `weekday` - Encoded using `OneHotEncoder` because these features have no intrinsic order
- **Ordinal Features**: No ordinal features
- **Passthrough Features**: No passthrough features

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Ridge Regression**: Linear regression with L2 regularization because it includes a regularization term to prevent overfitting. Since different input features will have varying levels of influence on the resale price, the reduction of coefficient size caused by the shrinkage penalty translates to each predictor feature having less influence on the final prediction
2. **Random Forest**: Non-linear model that averages many trees. It can capture effects that aren’t straight lines and interactions between features. It’s also fairly robust to outliers and needs little tuning. Averaging many deep trees reduces overfitting, so it’s a safe non-linear option.
3. **Gradient boosting**: a non-linear model that builds trees one after another.
Each new tree corrects the errors of the previous ones.

### Hyperparameter Tuning
- Grid search is performed for all models because tuning model parameters can achieve the best possible performance. Grid Search CV's process is exhaustive but it ensures the best combination
- Parameters tuned cover a wide range because the right combination of these settings ensures balance and should perform well on unseen data
- 5-fold cross-validation is used during tuning because averaged across all 5 folds will give a more reliable estimate of model performance.

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual prices
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same unit as the target
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model

## Deployment Considerations
- Code is structured into modular well-organized componenets making code easier to maintain, more scalable, and ready for real world applications