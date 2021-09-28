from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from regression_model.processing import preprocessors as pp
from regression_model.processing import features
from regression_model.config import config

import logging


_logger = logging.getLogger(__name__)

pm_pipe = Pipeline(
    [
        (
            "date_transformer",
            features.DateTransformer(variables=config.DATE_FEATURES),
        ),
        (
            "nlp_transformer",
            features.NLPTransformer(variables=config.NLP_FEATURES),
        ),
        (
            "drop_features",
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES),
        ),
        ("XGB_model", XGBRegressor(random_state=8)),
    ]
)
