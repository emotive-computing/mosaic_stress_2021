import os
import sys
import pandas as pd
from tensorflow import keras
from .stage_base import StageBase

# TODO: add LoadTrainData and LoadTestData Classes, push to c as "train_data" and "test_data"

class DataLoaderStageBase(StageBase):
    def __init__(self):
        super().__init__()
        self.setLoggingPrefix('DataLoaderStage: ')
        return

    def _validate(self, dc):
        raise NotImplementedError()
        return

    def _execute(self, dc):
        raise NotImplementedError()
        return


class CSVDataLoaderStage(DataLoaderStageBase):
    def __init__(self):
        super().__init__()
        self._file_path = None

    def setFilePath(self, file_path):
        self._file_path = file_path
        return

    def _validate(self, dc):
        if self._file_path is None:
            raise RuntimeError("Need to call setFilePath() first")
        elif not os.path.exists(self._file_path):
            self.logError("File path does not exist: {}".format(self._file_path))
        return

    def _execute(self, dc):
        self.logInfo("Loading CSV data from: {}".format(self._file_path))
        dc.set_item('data_file_path', self._file_path)
        df = pd.read_csv(self._file_path)
        dc.set_item('data', df)
        return dc


class ObjectDataLoaderStage(DataLoaderStageBase):
    def __init__(self):
        super().__init__()
        self._data = None

    def setDataObject(self, data, reset_index=False):
        if reset_index:
            self._data = data.reset_index()
        else:
            self._data = data
        return

    def _validate(self, dc):
        if self._data is None:
            raise RuntimeError("Need to call setDataObject() first")
        return

    def _execute(self, dc):
        self.logInfo("Loading data from object in memory")
        dc.set_item('data', self._data)
        return dc


# TODO - proper handling of loading training/validation/testing data
class FashionMNISTLoaderStage(DataLoaderStageBase):
    def __init__(self):
        super().__init__()
        return

    def _validate(self, dc):
        return

    def _execute(self, dc):
        self.logInfo("Loading fashion_mnist dataset")
        fashion_mnist = keras.datasets.fashion_mnist
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        # make validation set, scale vals to range from 0-1
        X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
        y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
        X_test = X_test / 255.0
        data = {
            'X_valid': X_valid,
            'X_train': X_train,
            'y_valid': y_valid,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        dc.set_item('data', data)
        return dc
