import numpy as np
from sklearn.model_selection import train_test_split

from regression_model import pipeline
from regression_model.processing.data_management import load_dataset, save_pipeline
from regression_model.config import config
from regression_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # drop rows with null target
    #data.dropna(axis=0, subset=[config.TARGET], inplace=True) moved to data_management

    # Only null vals are in selftext col at this point, replacing with empty str
    #data.fillna("", inplace=True) moved to data_management

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], train_size=0.8, test_size=0.2, random_state=8
    )

    pipeline.pm_pipe.fit(X_train[config.FEATURES], y_train)

    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.pm_pipe)


if __name__ == "__main__":
    run_training()
