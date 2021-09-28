import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import spacy

from regression_model.processing.errors import InvalidModelInputError


# may need clean up for different date formats
class DateTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            
            # Convert date column into datetime format
            X[feature] = pd.to_datetime(X[feature], unit='s')

            # Split datetime into separate columns
            X["day"] = X[feature].map(lambda x: x.day)
            X["month"] = X[feature].map(lambda x: x.month)
            X["year"] = X[feature].map(lambda x: x.year)
            X["time"] = X[feature].map(lambda x: x.time())

        return X


class NLPTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, pipe='en_core_web_lg'):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.nlp = spacy.load(pipe)

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            
            with self.nlp.disable_pipes():
                vectors = np.array([self.nlp(text).vector for text in X[feature]])
                vectors_df = pd.DataFrame(vectors, columns=[str(feature)+str(i) for i in range(vectors.shape[1])])
                vectors_df.index = X.index
                X = pd.concat([X, vectors_df], axis=1)

        return X