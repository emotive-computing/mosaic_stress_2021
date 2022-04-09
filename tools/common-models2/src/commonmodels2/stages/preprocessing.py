import os
import sys
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from .stage_base import StageBase

class PreprocessingStageBase(StageBase, ABC):
    def __init__(self):
        super().__init__()
        self.setLoggingPrefix('PreprocessingStage: ')
        self._fit_transform_data_idx = None
        self._transform_data_idx = None
        return

    def set_fit_transform_data_idx(self, fit_transform_data_idx):
        self._fit_transform_data_idx = fit_transform_data_idx
        return

    def set_transform_data_idx(self, transform_data_idx):
        self._transform_data_idx = transform_data_idx
        return

    def _validate(self, dc):
        try:
            cv_splits = dc.get_item('cv_splits')
        except:
            if self._fit_transform_data_idx is None:
                raise ValueError("set_fit_transform_data_idx() must be called for preprocessing stages")
            if self._transform_data_idx is None:
                raise ValueError("set_transform_data_idx() must be called for preprocessing stages")
        return
    
    def _execute(self, dc):
        try:
            cv_splits = dc.get_item('cv_splits')
            if self._fit_transform_data_idx is None:
                self._fit_transform_data_idx = cv_splits[0]
            if self._transform_data_idx is None:
                self._transform_data_idx = cv_splits[1]
        except:
            self.logInfo("Preprocessing stage will be applied to entire data set.  Are you sure this is right?")
            self._fit_transform_data_idx = list(range(dc.get_item('data').shape[0]))
            self._transform_data_idx = None
        return dc


class ImputerPreprocessingStage(PreprocessingStageBase):
    _valid_imputers = ['mean', 'median', 'most_frequent', 'constant']

    def __init__(self, cols, strategy, fill_value=None):
        super().__init__()
        self._cols = cols
        self._strategy = strategy.lower()
        self._fill_value = fill_value
        return

    def _get_imputer(self):
        return SimpleImputer(strategy=self._strategy, fill_value=self._fill_value)
        
    def _validate(self, dc):
        super()._validate(dc)
        if self._strategy not in self._valid_imputers:
            raise ValueError("Unknown strategy passed to {}".format(type(self).__name__))
        return

    def _execute(self, dc):
        dc = super()._execute(dc)
        imputer = self._get_imputer()
        self.logInfo("Imputing missing values for columns: {}".format(self._cols))
        X = dc.get_item('data')
        data_to_impute = X[self._cols]

        fit_transform_data = data_to_impute.iloc[self._fit_transform_data_idx, :]
        imputer = imputer.fit(fit_transform_data)
        fit_transform_data = imputer.transform(fit_transform_data)
        X.loc[self._fit_transform_data_idx, self._cols] = fit_transform_data

        if self._transform_data_idx:
            transform_data = data_to_impute.iloc[self._transform_data_idx, :]
            transform_data = imputer.transform(transform_data)
            X.loc[self._transform_data_idx, self._cols] = transform_data

        dc.set_item('data', X)
        return dc


class FeatureScalerPreprocessingStage(PreprocessingStageBase):
    _scalers = {
        'min-max': lambda feat_range: MinMaxScaler(feature_range=feat_range),
        'standardize': lambda dummy_param: StandardScaler()
    }

    def __init__(self, cols, strategy, feature_range=(0, 1)):
        super().__init__()
        self._cols = cols
        self._strategy = strategy.lower()
        self._feature_range = feature_range
        return

    def _get_scaler(self):
        self.logInfo("Scaler strategy selected as {}".format(self._strategy))
        return self._scalers[self._strategy](self._feature_range)

    def _validate(self, dc):
        super()._validate(dc)
        if self._strategy not in self._scalers.keys():
            raise ValueError("Unknown strategy passed to {}; must be one of {}".format(type(self).__name__, self._scalers.keys()))
        return

    def _execute(self, dc):
        dc = super()._execute(dc)
        scaler = self._get_scaler()
        self.logInfo("Scaling values for columns: {}".format(self._cols))
        X = dc.get_item('data')
        data_to_scale = X[self._cols]

        fit_transform_data = data_to_scale.iloc[self._fit_transform_data_idx, :]
        scaler = scaler.fit(fit_transform_data)
        fit_transform_data = scaler.transform(fit_transform_data)
        X.loc[self._fit_transform_data_idx, self._cols] = fit_transform_data

        if self._transform_data_idx:
            transform_data = data_to_scale.iloc[self._transform_data_idx, :]
            transform_data = scaler.transform(transform_data)
            X.loc[self._transform_data_idx, self._cols] = transform_data

        dc.set_item('data', X)
        return dc


class EncoderPreprocessingStage(PreprocessingStageBase):
    class Encoder:
        @abstractmethod
        def fit(self, data):
            return self

        @abstractmethod
        def transform(self, data):
            return data

    def __init__(self, cols, encoder):
        super().__init__()
        self._cols = cols
        self._encoder = encoder
        return

    def _get_encoder(self):
        if isinstance(self._encoder, str):
            return self._encoders[self._encoder]
        else:
            return self._encoder

    # TODO - fix one-hot encoding. Needs fit_transform and fit methods, perhaps in a strategy DP class?
    #@classmethod
    #def _one_hot_encode(cls, df, cols):
    #    for col in cols:
    #        encoder = OneHotEncoder()
    #        col_to_encode = df[[col]].astype('category').categorize()
    #        encoded_cols = encoder.fit_transform(col_to_encode)
    #        for c in encoded_cols:
    #            df[c] = encoded_cols[c]
    #        df = df.drop(col, axis=1)
    #    return df

    #@classmethod
    #def _label_encode(cls, df, cols):
    #    for col in cols:
    #        encoder = LabelEncoder()
    #        encoded_col = encoder.fit_transform(df[col])
    #        df[col] = encoded_col
    #    return df

    def _validate(self, dc):
        #super()._validate(dc)
        if isinstance(self._encoder, str) and self._encoder not in self._encoders.keys():
            raise ValueError("Unknown encoder string passed to {}; must be one of {}".format(type(self).__name__, self._encoders.keys()))
        # if not isinstance(self._encoder, type(self).Encoder):
        #     raise ValueError("Unknown encoder object passed to {}; must inherit from {}".format(type(self).__name__, self._encoder))
        return

    def _execute(self, dc):
        dc = super()._execute(dc)
        encoder = self._get_encoder()
        self.logInfo("Encoding labels for columns: {}".format(self._cols))
        X = dc.get_item('data')
        data_to_encode = X[self._cols]

        fit_transform_data = data_to_encode.iloc[self._fit_transform_data_idx, :]
        encoder = encoder.fit(fit_transform_data)
        fit_transform_data = encoder.transform(fit_transform_data)
        if len(fit_transform_data.shape) == 1:
            fit_transform_data = np.expand_dims(fit_transform_data, axis=1)
        if fit_transform_data.shape[1] == len(self._cols):
            X.loc[self._fit_transform_data_idx, self._cols] = fit_transform_data
        else:
            X = pd.concat([X, fit_transform_data], axis=1)
            X.loc[self._fit_transform_data_idx, fit_transform_data.columns] = fit_transform_data

        if self._transform_data_idx:
            transform_data = data_to_encode.iloc[self._transform_data_idx, :]
            transform_data = encoder.transform(transform_data)
            if transform_data.shape[1] == len(self._cols):
                X.loc[self._transform_data_idx, self._cols] = transform_data
            else:
                X.loc[self._transform_data_idx, transform_data.columns] = transform_data

        dc.set_item('data', X)
        return dc

EncoderPreprocessingStage._encoders = {
    #'onehotencoder': EncoderPreprocessingStage._one_hot_encode,
    #'labelencoder': EncoderPreprocessingStage._label_encode
    'labelencoder': LabelEncoder()
}
