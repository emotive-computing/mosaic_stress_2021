import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd
from inspect import signature
from tensorflow import keras
from skorch.net import NeuralNet
from abc import ABCMeta, abstractmethod
from commonmodels2.utils.utils import *
from commonmodels2.log.logger import Logger

class ModelBase(metaclass=ABCMeta):
    def __init__(self):
        self._model = None
        self._model_create_fn = None
        self._params = {}
        self._finalized = False

    def get_model(self):
        if self._finalized:
            return self._model
        else:
            raise RuntimeError("Models must be finalized prior to getting")

    def set_model_create_func(self, func):
        sig = signature(func)
        if len(sig.parameters) != 1:
            raise ValueError("model_create_fn must accept a single argument")
        param = [v for v in sig.parameters.values()][0]
        if not (param.kind == param.POSITIONAL_ONLY or param.kind == param.POSITIONAL_OR_KEYWORD):
            raise ValueError("model_create_fn must have similar prototype to `def func(params):`")
        if not (param.default is param.empty):
            raise ValueError("model_create_fn argument cannot have default value")
        self._model_create_fn = func

    def get_params(self):
        return copy.deepcopy(self._params)

    def set_params(self, params):
        self._params = copy.deepcopy(params)

    def load(self, file_path):
        if not os.path.isdir(os.path.dirname(file_path)):
            raise ValueError("Cannot load model from '%s' because file does not exist"%(file_path))

    def save(self, out_folder, file_name):
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

    def finalize(self):
        self._model = self._model_create_fn(self._params)
        self._finalized = True

    def fit(self, X, y):
        if not self._finalized:
            raise RuntimeError("Models must be finalized prior to fitting")

    def predict(self, X):
        if not self._finalized:
            raise RuntimeError("Models must be finalized prior to prediction")

    params = property(get_params, set_params)


class SklearnModel(ModelBase):
    def __init__(self):
        super().__init__()
        self._model_params = None

    def set_params(self, params):
        if params.get('model'):
            self.set_model_params(params['model'])
    
    def get_model_params(self):
        return self._model_params

    def set_model_params(self, model_params):
        self._model_params = model_params

    def load(self, file_path):
        super().load(file_path)
        with open(file_path, "rb") as in_file:
            self._model = pickle.load(out_file)
        self._finalized = True

    def save(self, out_folder, file_name):
        super().save(out_folder, file_name)
        if not file_name.endswith('.pkl'):
            if '.' in file_name:
                file_name = file_name.split('.')[0]+'.pkl'
            else:
                file_name = file_name+'.pkl'
        file_path = os.path.join(out_folder, file_name)
        with open(file_path, "wb") as out_file:
            pickle.dump(self._model, out_file)

    def finalize(self):
        self.params = self.model_params
        super().finalize()

    def fit(self, X, y):
        super().fit(X, y)
        self._model.fit(X, y)
        return self._model

    def predict(self, X):
        super().predict(X)
        preds = self._model.predict(X)
        return preds

    model_params = property(get_model_params, set_model_params)

class TensorFlowModel(ModelBase):
    def __init__(self):
        super().__init__()
        self._model_params = {}
        self._compile_params = {}
        self._predict_params = {}
        self._fit_params = {}
        self._fit_transformer_fn = None
        self._pred_transformer_fn = None

    def set_params(self, params):
        if params.get('model'):
            self.set_model_params(params['model'])
        if params.get('fit'):
            self.set_fit_params(params['fit'])
        if params.get('compile'):
            self.set_optimizer_params(params['compile'])
        if params.get('predict'):
            self.set_predict_params(params['predict'])

    def get_model_params(self):
        return self._model_params

    def set_model_params(self, model_params):
        self._model_params = model_params

    def get_fit_params(self):
        return self._fit_params

    def set_fit_params(self, fit_params):
        self._fit_params = fit_params

    def get_compile_params(self):
        return self._compile_params

    def set_compile_params(self, compile_params):
        self._compile_params = compile_params    

    def get_predict_params(self):
        return self._predict_params

    def set_predict_params(self, predict_params):
        self._predict_params = predict_params

    def set_fit_transformer(self, data_trans_func):
        sig = signature(data_trans_func)
        if len(sig.parameters) != 2:
            raise ValueError("data_transformer_fn must accept two arguments: X (data) and y (labels)")
        for param_idx in range(2):
            param = [v for v in sig.parameters.values()][param_idx]
            if not (param.kind == param.POSITIONAL_ONLY or param.kind == param.POSITIONAL_OR_KEYWORD):
                raise ValueError("data_transformer_fn must have similar prototype to `def func(X, y):`")
        if not (param.default is param.empty):
            raise ValueError("data_transformer_fn argument cannot have default value")
        self._fit_transformer_fn = data_trans_func

    def set_prediction_transformer(self, pred_trans_func):
        sig = signature(pred_trans_func)
        if len(sig.parameters) != 1:
            raise ValueError("pred_transformer_fn must accept one argument: y_pred (labels)")
        for param_idx in range(1):
            param = [v for v in sig.parameters.values()][param_idx]
            if not (param.kind == param.POSITIONAL_ONLY or param.kind == param.POSITIONAL_OR_KEYWORD):
                raise ValueError("pred_transformer_fn must have similar prototype to `def func(y_pred):`")
        if not (param.default is param.empty):
            raise ValueError("pred_transformer_fn argument cannot have default value")
        self._pred_transformer_fn = pred_trans_func

    def load(self, file_path):
        super().load(file_path)
        self._model = keras.models.load_model(file_path)
        self._finalized = True

    def save(self, out_folder, file_name):
        super().save(out_folder, file_name)
        file_path = os.path.join(out_folder, file_name)
        self._model.save(file_path)

    def finalize(self):
        self.params = self.model_params
        super().finalize()
        try:
            self._model.compile(**self.compile_params) 
        except Exception as e:
            raise RuntimeError("Failed to compile TensorFlowModel with exception: {}".format(str(e)))

    def fit(self, X, y):
        super().fit(X, y)
        if self._fit_transformer_fn is not None:
            trans_X, trans_y = self._fit_transformer_fn(X, y)
            self._model.fit(trans_X, trans_y, **self._fit_params) 
        else:
            self._model.fit(X, y, **self._fit_params) 
        return self._model

    def predict(self, X):
        super().predict(X)
        if self._fit_transformer_fn is not None:
            trans_X, trans_y = self._fit_transformer_fn(X, None)
            preds = self._model.predict(trans_X, **self._predict_params)
        else:
            preds = self._model.predict(X, **self._predict_params)

        if self._pred_transformer_fn is not None:
            preds = self._pred_transformer_fn(preds)

        return preds

    model_params = property(get_model_params, set_model_params)
    fit_params = property(get_fit_params, set_fit_params)
    compile_params = property(get_compile_params, set_compile_params)
    predict_params = property(get_predict_params, set_predict_params)      

class PyTorchModel(ModelBase):
    def __init__(self):
        super().__init__()
        self._model_params = None
        self._optimizer_params = None
        self._criterion_params = None
        self._fit_params = None
        self._fit_transformer_fn = PyTorchModel._default_fit_transformer
        self._pred_transformer_fn = None
    
    def set_params(self, params):
        if params.get('model'):
            self.set_model_params(params['model'])
        if params.get('fit'):
            self.set_fit_params(params['fit'])
        if params.get('optimizer'):
            self.set_optimizer_params(params['optimizer'])
        if params.get('criterion'):
            self.set_criterion_params(params['criterion'])

    def get_criterion_params(self, fn_key="criterion"):
        criterion_params = copy.deepcopy(self._criterion_params)
        if criterion_params:
            criterion_fn = criterion_params.pop(fn_key,None)
            if not criterion_fn: 
                raise ValueError(f"Requires a {fn_key} parameter")
            criterion_fn = get_torch_loss_func(criterion_fn)
            criterion_params = {f"criterion__{k}":v for k,v in criterion_params.items()}
        else:
            raise ValueError(f"Requires a {fn_key} parameter")
        return criterion_fn, criterion_params

    def set_criterion_params(self, criterion_params):
        self._criterion_params = criterion_params

    def get_model_params(self):
        return self._model_params

    def set_model_params(self, model_params):
        self._model_params = model_params

    def get_optimizer_params(self, fn_key="optimizer", default_fn="sgd"):
        optim_params = copy.deepcopy(self._optimizer_params)
        if optim_params:
            optim_fn = optim_params.pop(fn_key, default_fn)
            optim_fn = get_torch_optimizer(optim_fn)
            optim_params = {f"optimizer__{k}":v for k,v in optim_params.items()}
        else:
            optim_fn = get_torch_optimizer(default_fn)
            optim_params = {}
        return optim_fn, optim_params

    def set_optimizer_params(self, optimizer_params):
        self._optimizer_params = optimizer_params

    def get_fit_params(self):
        return self._fit_params

    def set_fit_params(self, fit_params):
        self._fit_params = fit_params
    
    def compile(self):
        loss_fn, loss_params = self.get_criterion_params()
        optim_fn, optim_params = self.get_optimizer_params()
        fit_params = self.get_fit_params()
        user_params = {}
        if loss_params:
            user_params.update(loss_params)
        if optim_params:
            user_params.update(optim_params)
        if fit_params:
            user_params.update(fit_params)
        self._model = NeuralNet(self._model,loss_fn,optimizer=optim_fn,**user_params)

    @classmethod
    def _default_fit_transformer(cls, X, y):
        new_X = X
        if isinstance(X,pd.DataFrame):
            # BB - pytorch doesn't like using float64 tensors (X) with float64 labels (y), so
            #      stick to float32 for now
            #best_type = np.float64 if np.any(X.dtypes == np.float64) else np.float32
            best_type = np.float32
            new_X = X.to_numpy(dtype=best_type)
        elif isinstance(X,np.ndarray):
            best_type = X.dtype
            new_X = X.astype(best_type)

        new_y = y
        if isinstance(y,pd.DataFrame):
            best_type = np.float64 if np.any(y.dtypes == np.float64) else np.float32
            new_y = y.to_numpy(dtype=best_type)
        elif isinstance(y, pd.Series):
            best_type = y.dtype
            new_y = y.to_numpy(dtype=best_type)
        elif isinstance(y, np.ndarray):
            best_type = y.dtype
            new_y = y.astype(best_type)
        elif isinstance(y, list):
            best_type = np.float64 if np.any([type(elem) == np.float64 for elem in y]) else np.float32
            new_y = np.array(y).astype(best_type)

        return new_X, new_y

    def set_fit_transformer(self, data_trans_func):
        sig = signature(data_trans_func)
        if len(sig.parameters) != 2:
            raise ValueError("data_transformer_fn must accept two arguments: X (data) and y (labels)")
        for param_idx in range(2):
            param = [v for v in sig.parameters.values()][param_idx]
            if not (param.kind == param.POSITIONAL_ONLY or param.kind == param.POSITIONAL_OR_KEYWORD):
                raise ValueError("data_transformer_fn must have similar prototype to `def func(X, y):`")
        if not (param.default is param.empty):
            raise ValueError("data_transformer_fn argument cannot have default value")
        self._fit_transformer_fn = data_trans_func

    def set_prediction_transformer(self, pred_trans_func):
        sig = signature(pred_trans_func)
        if len(sig.parameters) != 1:
            raise ValueError("pred_transformer_fn must accept one argument: y_pred (labels)")
        for param_idx in range(1):
            param = [v for v in sig.parameters.values()][param_idx]
            if not (param.kind == param.POSITIONAL_ONLY or param.kind == param.POSITIONAL_OR_KEYWORD):
                raise ValueError("pred_transformer_fn must have similar prototype to `def func(y_pred):`")
        if not (param.default is param.empty):
            raise ValueError("pred_transformer_fn argument cannot have default value")
        self._pred_transformer_fn = pred_trans_func

    def load(self, file_path):
        super().load(file_path)
        with open(file_path, "rb") as in_file:
            self._model = pickle.load(out_file)
        self._finalized = True

    def save(self, out_folder, file_name):
        super().save(out_folder, file_name)
        if not file_name.endswith('.pkl'):
            if '.' in file_name:
                file_name = file_name.split('.')[0]+'.pkl'
            else:
                file_name = file_name+'.pkl'
        file_path = os.path.join(out_folder, file_name)
        with open(file_path, "wb") as out_file:
            pickle.dump(self._model, out_file)

    def finalize(self):
        self.params = self.model_params
        super().finalize()
        try:
            self.compile() 
        except Exception as e:
            raise RuntimeError("Failed to compile PyTorchModel with exception: {}".format(str(e)))

    def fit(self, X, y):
        super().fit(X,y)
        if self._fit_transformer_fn is not None:
            trans_X, trans_y = self._fit_transformer_fn(X, y)
            self._model.fit(trans_X, trans_y)
        else:
            self._model.fit(X, y)
        return self._model

    def predict(self, X):
        super().predict(X)
        if self._fit_transformer_fn is not None:
            trans_X, trans_y = self._fit_transformer_fn(X, None)
            preds = self._model.predict(trans_X)
        else:
            preds = self._model.predict(X)

        if self._pred_transformer_fn is not None:
            preds = self._pred_transformer_fn(preds)

        return preds

    model_params = property(get_model_params, set_model_params)
    fit_params = property(get_fit_params, set_fit_params)
    optimizer_params = property(get_optimizer_params, set_optimizer_params)
    criterion_params = property(get_criterion_params, set_criterion_params)
