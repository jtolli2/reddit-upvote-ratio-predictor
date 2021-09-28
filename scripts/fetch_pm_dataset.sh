#!/usr/bin/env bash

kaggle datasets download -d jeffreytolliver/paper-mario-reddit-submissions-20210724 -p packages/regression_model/regression_model/datasets/
unzip packages/regression_model/regression_model/datasets/* -d packages/regression_model/regression_model/datasets/