import os
import sys
import copy
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections.abc import Iterable
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder, Binarizer
from sklearn.impute import SimpleImputer
from .stage_base import StageBase
from ..log.logger import Logger

class PreprocessingStageBase(StageBase, ABC):
    def __init__(self):
        super().__init__()
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
        if self._fit_transform_data_idx is not None:
            if self._transform_data_idx is None:
                if len(self._fit_transform_data_idx) != dc.get_item('data').shape[0]:
                    Logger.getInst().warning("set_fit_transform_data_idx() was called indexing a subset of the data, but set_transform_data_idx() has not been called.  Are you sure this is right?")
        else:
            try:
                cv_splits = dc.get_item('cv_splits')
                if len(cv_splits) != 1:
                    raise ValueError("Too many CV folds detected.  Are you running this outside of a nested CV stage?  Make sure only one fold is available, or manually call set_fit_transform_data_idx() and set_transform_data_idx()")
                if len(cv_splits[0]) != 2:
                    raise ValueError("The CV fold needs to be a tuple of (train_fold_idx, test_fold_idx)")
            except:
                Logger.getInst().warning('set_fit_transform_data_idx() was not called. Applying this preprocessing step to the entire dataset.')
                self._fit_transform_data_idx = range(dc.get_item('data').shape[0])

        return
    
    def _execute(self, dc):
        if self._fit_transform_data_idx is None:
            cv_splits = dc.get_item('cv_splits')
            if self._fit_transform_data_idx is None:
                self._fit_transform_data_idx = cv_splits[0][0]
            if self._transform_data_idx is None:
                self._transform_data_idx = cv_splits[0][1]
        return dc


class ImputerPreprocessingStage(PreprocessingStageBase):
    _valid_imputers = ['mean', 'median', 'most_frequent', 'constant']

    def __init__(self, cols, strategy, fill_value=None):
        super().__init__()
        if isinstance(cols, pd.DataFrame):
            self._cols = cols.values.flatten()
        elif isinstance(cols, np.ndarray):
            self._cols = cols.flatten()
        else:
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
        Logger.getInst().info("Imputing missing values for columns: {}".format(self._cols))
        X = dc.get_item('data')
        data_to_impute = X[self._cols]

        fit_transform_data = data_to_impute.iloc[self._fit_transform_data_idx, :]
        imputer = imputer.fit(fit_transform_data)
        fit_transform_data = imputer.transform(fit_transform_data)
        X.loc[self._fit_transform_data_idx, self._cols] = fit_transform_data

        if self._transform_data_idx is not None:
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
        if isinstance(cols, pd.DataFrame):
            self._cols = cols.values.flatten()
        elif isinstance(cols, np.ndarray):
            self._cols = cols.flatten()
        else:
            self._cols = cols
        self._strategy = strategy.lower()
        self._feature_range = feature_range
        return

    def _get_scaler(self):
        Logger.getInst().info("Scaler strategy selected as {}".format(self._strategy))
        return self._scalers[self._strategy](self._feature_range)

    def _validate(self, dc):
        super()._validate(dc)
        if self._strategy not in self._scalers.keys():
            raise ValueError("Unknown strategy passed to {}; must be one of {}".format(type(self).__name__, self._scalers.keys()))
        return

    def _execute(self, dc):
        dc = super()._execute(dc)
        scaler = self._get_scaler()
        Logger.getInst().info("Scaling values for columns: {}".format(self._cols))
        X = dc.get_item('data')
        data_to_scale = X[self._cols]

        fit_transform_data = data_to_scale.iloc[self._fit_transform_data_idx, :]
        scaler = scaler.fit(fit_transform_data)
        fit_transform_data = scaler.transform(fit_transform_data)
        X.loc[self._fit_transform_data_idx, self._cols] = fit_transform_data

        if self._transform_data_idx is not None:
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

    def __init__(self, cols, encoder, encoder_args=None):
        super().__init__()
        if isinstance(cols, pd.DataFrame):
            self._cols = cols.values.flatten()
        elif isinstance(cols, np.ndarray):
            self._cols = cols.flatten()
        else:
            self._cols = cols
        self._encoder = encoder
        self._encoder_args = encoder_args
        return

    def _get_encoder(self):
        if isinstance(self._encoder, str):
            if self._encoder_args is None:
                return self._encoders[self._encoder]()
            else:
                return self._encoders[self._encoder](**self._encoder_args)
        else:
            return self._encoder

    class BinarizeMedianEncoder(Encoder):
        def __init__(self, group_by=None):
            self._group_by = group_by

        def fit(self, data):
            return self # Nothing to do

        def transform(self, data):
            if len(data.shape) == 1:
                if self._group_by is not None:
                    raise ValueError("group_by is set for {} but only one column of data is present for binarization".format(type(self).__name__))
                median_val = np.median(data)
                binarizer = Binarizer(threshold=median_val)
                trans_data = binarizer.transform(data)
            else:
                if self._group_by is not None:
                    trans_data = copy.deepcopy(data)
                    for group_id, group_df in trans_data.groupby(by=self._group_by, axis=0, as_index=False, sort=False):
                        for col_idx in np.where(group_df.columns != self._group_by):
                            if not isinstance(col_idx, Iterable):
                                col_idx = [col_idx] # A list indexer for a DF will return a DF, not a series
                            group_df_col = group_df.iloc[:,col_idx]
                            median_val = np.nanmedian(group_df_col)
                            binarizer = Binarizer(threshold=median_val, copy=False)
                            bin_trans_data = binarizer.transform(group_df_col)
                            trans_data.loc[group_df.index, group_df_col.columns[0] == trans_data.columns] = bin_trans_data
                else:
                    trans_data = copy.deepcopy(data)
                    for col_idx in range(trans_data.shape[1]):
                        if not isinstance(col_idx, Iterable):
                            col_idx = [col_idx] # A list indexer for a DF will return a DF, not a series
                        trans_data_col = trans_data.iloc[:,col_idx]
                        median_val = np.nanmedian(trans_data_col)
                        binarizer = Binarizer(threshold=median_val, copy=False)
                        bin_trans_data = binarizer.transform(trans_data_col)
                        trans_data.iloc[:,col_idx] = bin_trans_data

            for col in trans_data.columns:
                if col != self._group_by:
                    trans_data[col] = trans_data[col].astype(int)
            return trans_data
        

    class OneHotEncoder(Encoder):
        def fit(self, data):
            return self

        def transform(self, data):
            encoded_dfs = []
            for col_idx in range(data.shape[1]):
                encoder = OneHotEncoder()
                col_to_encode = data.iloc[:,[col_idx]].astype('category')
                encoded_cols = encoder.fit_transform(col_to_encode)
                encoded_df = pd.DataFrame(data=encoded_cols.toarray(), columns=encoder.get_feature_names_out())
                encoded_dfs.append(encoded_df)

            encoded_data = pd.concat(encoded_dfs, axis=1)
            return encoded_data

    def _validate(self, dc):
        super()._validate(dc)
        if isinstance(self._encoder, str) and self._encoder not in self._encoders.keys():
            raise ValueError("Unknown encoder string passed to {}; must be one of {}".format(type(self).__name__, self._encoders.keys()))
        # if not isinstance(self._encoder, type(self).Encoder):
        #     raise ValueError("Unknown encoder object passed to {}; must inherit from {}".format(type(self).__name__, self._encoder))
        return

    def _execute(self, dc):
        dc = super()._execute(dc)
        encoder = self._get_encoder()
        X = dc.get_item('data')
        data_to_encode = X[self._cols]
        Logger.getInst().info("Encoding labels for columns: {}".format(self._cols))

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
    'label_encoder': LabelEncoder,
    'one_hot': EncoderPreprocessingStage.OneHotEncoder,
    'binarize_median': EncoderPreprocessingStage.BinarizeMedianEncoder
}
