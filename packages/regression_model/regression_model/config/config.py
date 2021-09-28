import pathlib

import regression_model

import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TESTING_DATA_FILE = 'test.csv'
TRAINING_DATA_FILE = 'pm_data.csv'
TARGET = 'upvote_ratio'


# variables
FEATURES = [
    'title',
    'selftext',
    #'score',#maybe create product with target
    #'upvote_ratio',#the target
    #'total_awards_received',#maybe create product with target
    'subreddit_subscribers',#possible leakage
    'spoiler',
    'over_18',
    #'preview', high % missing
    #'link_flair_css_class', high % missing
    'created_utc',
    #'author_flair_css_class', high % missing
    'is_original_content',
]

NLP_FEATURES = [
    'title',
    'selftext',
]

DATE_FEATURES = [
    'created_utc',
]

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = [
    'title',
    'selftext',
    'created_utc',
    'time',# have to look into converting this obj to use in model
]

'''
# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['LotFrontage']

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = [
    'MasVnrType',
    'BsmtQual',
    'BsmtExposure',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
]

TEMPORAL_VARS = 'YearRemodAdd'

# variables to log transform
NUMERICALS_LOG_VARS = ['LotFrontage', '1stFlrSF', 'GrLivArea']

# categorical variables to encode
CATEGORICAL_VARS = [
    #'link_flair_css_class', high % missing
    #'author_flair_css_class', high % missing
]

NUMERICAL_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA
]
'''

PIPELINE_NAME = 'reddit_regression'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05
