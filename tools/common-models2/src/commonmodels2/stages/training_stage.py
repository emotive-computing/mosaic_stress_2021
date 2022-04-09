import numpy as np
import pandas as pd

from collections.abc import Mapping, Iterable
from .stage_base import StageBase
from .prediction_stage import PredictionContext
from ..models.model import ModelBase
from ..log.logger import Logger

class ModelTrainingStage(StageBase):
    def __init__(self, train_idx):
        super().__init__()
        self._train_idx = train_idx
        self._training_context = None
        return

    def setTrainingContext(self, training_context):
        self._training_context = training_context
        return

    def _validate(self, dc):
        if not isinstance(self._training_context, SupervisedTrainingContext):
            raise ValueError("{} requires a context of type {}".format(type(self).__name__, type(SupervisedTrainingContext)))
        if "data" not in dc.get_keys():
            raise ValueError("{} needs a dataframe object named 'data' to be present in the data container".format(type(self).__name__))
        if min(self._train_idx) < 0 or max(self._train_idx) >= dc.get_item("data").shape[0]:
            raise ValueError("Training indices exceed bounds of the data size in {}".format(type(self).__name__))
        return

    def _execute(self, dc):
        data = dc.get_item('data')
        Logger.getInst().info("Starting model training stage with model {}".format(type(self._training_context.model).__name__))

        data_train = data.iloc[self._train_idx, :]
        label_train = data_train[self._training_context.label_cols]
        data_train = data_train[self._training_context.feature_cols]

        if label_train.shape[1] == 1:
            label_train = label_train.values.flatten()

        self._training_context.model.finalize()
        fitted_model = self._training_context.model.fit(data_train, label_train)
        #dc.set_item('trained_model', fitted_model)
        dc.set_item('trained_model', self._training_context.model)
        return dc


class TrainingContext(PredictionContext):
    def __init__(self):
        super().__init__()
        self._model = None
        return

    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model
        return

    def validate(self):
        super().validate()
        if not isinstance(self._model, ModelBase):
            raise TypeError("set_model() must be called with an instance of ModelBase")
        return

    model = property(get_model, set_model)


class SupervisedTrainingContext(TrainingContext):
    def __init__(self):
        super().__init__()
        self._label_cols = None
        return

    def get_label_cols(self):
        return self._label_cols

    def set_label_cols(self, label_cols):
        if isinstance(label_cols, pd.DataFrame):
            self._label_cols = label_cols.values.flatten()
        elif isinstance(label_cols, np.ndarray):
            self._label_cols = label.flatten()
        elif isinstance(label_cols, str) or isinstance(label_cols, Iterable):
            self._label_cols = label_cols
        else:
            raise ValueError("label argument must be Iterable or string type")
        return

    def validate(self):
        super().validate()
        if not isinstance(self._label_cols, str) and not isinstance(self._label_cols, Iterable):
            raise TypeError("set_label_cols() must be initialized to Iterable or string type")
        return

    label_cols = property(get_label_cols, set_label_cols)
