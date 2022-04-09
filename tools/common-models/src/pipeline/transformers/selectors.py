import os

import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.configuration.settings_template import Settings
from src.io.read_data_input import Dataset
from src.models.nets.utils import sequence
from src.metrics.output_columns import PROBABILITY_COLUMN_NAME_SUFFIX


class WindowSelector(BaseEstimator, TransformerMixin):

    def __init__(self, cols, window_size, pad_sequences=False):
        self.cols = cols
        self.use_sliding_window_size = window_size
        self.pad_sequences = pad_sequences

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X = X.reset_index()
        X['x_idx'] = X.index
        d = {}
        Y = X.groupby([Settings.COLUMNS.ORDER_IN_GROUPS_BY_COLUMN]).apply(lambda x: x.sort_values(by=[Settings.COLUMNS.ORDER_IN_GROUPS_SORT_BY_COLUMN]))
        for id, curr in Y.groupby([Settings.COLUMNS.ORDER_IN_GROUPS_BY_COLUMN]):

            indexes = curr['x_idx'].values

            for i in range(1, len(indexes)+1):
                start = i - self.use_sliding_window_size
                start = start if start > 0 else 0

                a = indexes[start:i]

                d[a[len(a) - 1]] = a

        idxs = [d[x] for x in X.index]
        xs = [X.iloc[j, :] for j in idxs]
        vals = [np.array(i.loc[:, self.cols].values) for i in xs]

        if self.pad_sequences:
            vals = sequence.pad_sequences(vals, maxlen=self.use_sliding_window_size, padding='pre')
        #
        # for idx, val in enumerate(vals):
        #     if len(val) < self.use_sliding_window_size:
        #         p = self.use_sliding_window_size - len(val)
        #         p1 = [None for i in range(p)]
        #         a = basic_utils.concatenate(p1, val)
        #         vals[idx] = a

        return vals


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        res = np.array(X.loc[:,self.cols])
        if res.dtype != np.floating:
            raise ValueError("The array has non floating point data. Please make sure the columns selected have floating point data")
        return res


class TextSelector(BaseEstimator, TransformerMixin):

    def __init__(self, col):
        self.col = col

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X[self.col].values

class PredictedItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, model, feature_source):
        self.cols = Settings.COLUMNS.Y_LABELS_TO_PREDICT
        self.prev_model_name = model.get_prev_class_name()
        self.feature_source = feature_source

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        predicted_cols = []
        for col in self.cols:
            predictions = pd.read_csv(os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, "predictions", self.prev_model_name, "{}-{}-predictions.csv".format(col, str(self.feature_source.__name__))))
            result = pd.merge(X, predictions, on=Settings.COLUMNS.IDENTIFIER, how='left')
            if len(result.values) != len(X.values):
                raise RuntimeError("Bad")

            le = Dataset().get_saved_label_encoder(col, initialize_if_missing=True)
            true_probability_column_name = le.classes_[1] + PROBABILITY_COLUMN_NAME_SUFFIX
            predicted_cols.append(result[true_probability_column_name].values)

        return np.asarray(predicted_cols).T

