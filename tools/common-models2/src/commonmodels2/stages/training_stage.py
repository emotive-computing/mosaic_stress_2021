from collections.abc import Mapping, Iterable
from .stage_base import StageBase
from .prediction_stage import PredictionContext
from ..models.model import ModelBase

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
        self.logInfo("Starting model training stage with model {}".format(type(self._training_context.model).__name__))

        data_train = data.iloc[self._train_idx, :]
        label_train = data_train[self._training_context.y_label]
        data_train = data_train[self._training_context.feature_cols]

        if label_train.shape[1] == 1:
            label_train = label_train.values.flatten()

        self._training_context.model.finalize()
        fitted_model = self._training_context.model.fit(data_train, label_train)
        dc.set_item('trained_model', fitted_model)
        return dc

# This doesn't do anything. We can either update the different models to only hold the model 
# and use this as a Trainer (contains parameters for training and Model and fit function)
# Or get rid of this
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
            raise TypeError("model must be initialized to instance of ModelBase")
        return

    model = property(get_model, set_model)


class SupervisedTrainingContext(TrainingContext):
    def __init__(self):
        super().__init__()
        self._y_label = None
        return

    def get_y_label(self):
        return self._y_label

    def set_y_label(self, y_label):
        if isinstance(y_label, str) or isinstance(y_label, Iterable):
            self._y_label = y_label
        else:
            raise ValueError('label argument must be Iterable or string type')
        return

    def validate(self):
        super().validate()
        if not isinstance(self._y_label, str) and not isinstance(self._y_label, Iterable):
            raise TypeError("y_label must be initialized to Iterable or string type")
        return

    y_label = property(get_y_label, set_y_label)



