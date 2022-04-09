import os
import sys
import copy
import numpy as np
import pandas as pd
from inspect import signature
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

    def finalize(self):
        self.params = self.model_params
        super().finalize()
        try:
            self._model.compile(**self.compile_params) 
        except Exception as e:
            raise RuntimeError("Failed to compile TensorFlowModel with exception: {}".format(str(e)))

    def fit(self, X, y):
        super().fit(X, y)
        self._model.fit(X, y, **self._fit_params) 
        return self._model

    def predict(self, X):
        super().predict(X)
        preds = self._model.predict(X, **self._predict_params)
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

    def _transform_x_data(self, X):
        if isinstance(X,pd.DataFrame):
            data = X.to_numpy(dtype=np.float32)
        elif isinstance(X,np.ndarray):
            data = X.astype(np.float32)
        return data

    def _transform_y_data(self, y):
        if isinstance(y,pd.DataFrame):
            data = y.to_numpy(dtype=np.float32)
        elif isinstance(y, pd.Series):
            data = y.to_numpy(dtype=np.float32)
        elif isinstance(y, np.ndarray):
            data = y.astype(np.float32)
        elif isinstance(y, list):
            data = np.array(y).astype(np.float32)
        return data

    def finalize(self):
        self.params = self.model_params
        super().finalize()
        try:
            self.compile() 
        except Exception as e:
            raise RuntimeError("Failed to compile PyTorchModel with exception: {}".format(str(e)))

    def fit(self, X, y):
        # Pass in parameters for fit loop
        super().fit(X,y)
        X = self._transform_x_data(X)
        self._model.fit(X, y)
        Logger.getInst().info(f"Model Parameters: {self._model.get_params_for('module')}")
        Logger.getInst().info(f"Fit Parameters: {self._model.get_params_for('iterator')}")
        Logger.getInst().info(f"Optimizer Parameters: {self._model.get_params_for('optimizer')}")
        return self._model

    def predict(self, X):
        super().predict(X)
        X = self._transform_x_data(X)
        preds = self._model.predict(X)
        return preds

    model_params = property(get_model_params, set_model_params)
    fit_params = property(get_fit_params, set_fit_params)
    optimizer_params = property(get_optimizer_params, set_optimizer_params)
    criterion_params = property(get_criterion_params, set_criterion_params)
