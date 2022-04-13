import os
import copy
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from .stage_base import StageBase
from .pipeline import Pipeline
from ..models.model import ModelBase, PyTorchModel, SklearnModel, TensorFlowModel
from .load_data import DataFrameLoaderStage
from .preprocessing import PreprocessingStageBase
from .training_stage import ModelTrainingStage, SupervisedTrainingContext
from .evaluation_stage import EvaluationStage, SupervisedEvaluationContext
from .prediction_stage import ModelPredictionStage, PredictionContext
from ..utils.utils import get_sklearn_scoring_func, get_tensorflow_loss_func, get_torch_loss_func, get_any_scoring_func
from ..log.logger import Logger

class GenerateCVFoldsStage(StageBase):
    def __init__(self, strategy, strategy_args):
        super().__init__()
        self._strategy = strategy.lower()
        self._strategy_args = strategy_args

    @classmethod
    def _random_kfold(cls, data, args):
        if 'num_folds' not in args.keys():
            raise ValueError("{} with 'random' strategy must provide a 'num_folds' strategy arg".format(type(cls).__name__))
        if 'seed' not in args.keys():
            raise ValueError("{} with 'random' strategy must provide a 'seed' strategy arg".format(type(cls).__name__))

        kf = KFold(n_splits=args['num_folds'], shuffle=True, random_state=args['seed'])
        return [x for x in kf.split(data)]

    @classmethod
    def _random_grouped_kfold(cls, data, args):
        if 'num_folds' not in args.keys():
            raise ValueError("{} with 'random_grouped' strategy must provide a 'num_folds' strategy arg".format(type(cls).__name__))
        if 'seed' not in args.keys():
            raise ValueError("{} with 'random_grouped' strategy must provide a 'seed' strategy arg".format(type(cls).__name__))
        if 'group_by' not in args.keys():
            raise ValueError("{} with 'random_grouped' strategy must provide a 'group_by' strategy arg".format(type(cls).__name__))
        data = shuffle(data, random_state=args['seed']).reset_index(drop=True) # Shuffle the data first because GroupKFold doesn't support it
        group_le = LabelEncoder()
        groups = group_le.fit_transform(data[args['group_by']].values)
        Logger.getInst().info("Shuffling and grouping data in folds by column: {}, found {} groups".format(args['group_by'], len(group_le.classes_)))
        gkf = GroupKFold(n_splits=args['num_folds'])
        return [x for x in gkf.split(data, groups=groups)]

    @classmethod
    def _stratified_kfold(cls, data, args):
        if 'num_folds' not in args.keys():
            raise ValueError("{} with 'stratified' strategy must provide a 'num_folds' strategy arg".format(type(cls).__name__))
        if 'percentile_bins' not in args.keys() and 'bin_edges' not in args.keys():
            raise ValueError("{} with 'stratified' strategy must provide either a 'percentile_bins' or 'bin_edges' strategy arg".format(type(cls).__name__))
        if 'seed' not in args.keys():
            Logger.getInst().info("{} with 'stratified' strategy can randomly shuffle the data if a 'seed' integer strategy arg is provided.  No shuffling used".format(type(cls).__name__))
        if 'stratify_on' not in args.keys():
            raise ValueError("{} with 'stratified' strategy must provide a 'stratify_on' strategy arg".format(type(cls).__name__))

        if 'seed' in args.keys():
            skf = StratifiedKFold(n_splits=args['num_folds'], shuffle=True, random_state=args['seed'])
        else:
            skf = StratifiedKFold(n_splits=args['num_folds'])

        if 'percentile_bins' in args.keys():
            y = pd.qcut(data[args['stratify_on']].argsort(kind='stable'), q=args['percentile_bins'], labels=range(args['percentile_bins'])).tolist()
        else:
            y = np.digitize(data[args['stratify_on']], bins=args['bin_edges'])
        return list(skf.split(data, y))

    @classmethod
    def _stratified_grouped_kfold(cls, data, args):
        if 'num_folds' not in args.keys():
            raise ValueError("{} with 'stratified_grouped' strategy must provide a 'num_folds' strategy arg".format(type(cls).__name__))
        if 'percentile_bins' not in args.keys() and 'bin_edges' not in args.keys():
            raise ValueError("{} with 'stratified_grouped' strategy must provide either a 'percentile_bins' or 'bin_edges' strategy arg".format(type(cls).__name__))
        if 'seed' not in args.keys():
            Logger.getInst().info("{} with 'stratified_grouped' strategy can randomly shuffle the data if a 'seed' integer strategy arg is provided.  No shuffling used".format(type(cls).__name__))
        if 'stratify_on' not in args.keys():
            raise ValueError("{} with 'stratified_grouped' strategy must provide a 'stratify_on' strategy arg".format(type(cls).__name__))
        if 'group_by' not in args.keys():
            raise ValueError("{} with 'stratified_grouped' strategy must provide a 'group_by' strategy arg".format(type(cls).__name__))

        group_le = LabelEncoder()
        groups = group_le.fit_transform(data[args['group_by']].values)
        grouped_data = data.groupby(by=groups, group_keys=True).mean()
        grouped_stratify_on = pd.Series(data[args['stratify_on']]).groupby(by=groups, group_keys=True).mean().values
        if 'seed' in args.keys():
            skf = StratifiedKFold(n_splits=args['num_folds'], shuffle=True, random_state=args['seed'])
        else:
            skf = StratifiedKFold(n_splits=args['num_folds'])

        if 'percentile_bins' in args.keys():
            grouped_stratify_on = pd.qcut(grouped_stratify_on.argsort(kind='stable'), q=args['percentile_bins'], labels=range(args['percentile_bins'])).tolist()
        else:
            grouped_stratify_on = np.digitize(grouped_stratify_on, bins=args['bin_edges'])
        train_test_index_list = skf.split(grouped_data, grouped_stratify_on)

        orig_train_test_index_list = []
        for train_idxs, test_idxs in train_test_index_list:
            train_group_ids = grouped_data.index[train_idxs]
            test_group_ids = grouped_data.index[test_idxs]
            orig_train_idxs = np.where(np.sum([groups == x for x in train_group_ids], axis=0) > 0)[0]
            orig_test_idxs = np.where(np.sum([groups == x for x in test_group_ids], axis=0) > 0)[0]

            orig_train_test_index_list.append((orig_train_idxs, orig_test_idxs))
        return orig_train_test_index_list

    @classmethod
    def _load_premade_folds(cls, data, args):
        if 'file_path' not in args.keys():
            raise ValueError("{} with 'load_premade' strategy must provide a 'file_path' strategy arg".format(type(cls).__name__))
        folds_df = pd.read_csv(args['file_path'])
        splits = []
        num_folds = int(folds_df.shape[1]/2)
        for fold_idx in range(1,num_folds+1):
            train_idxs = folds_df['Fold'+str(fold_idx)+'_train'].dropna().astype(int).values
            test_idxs = folds_df['Fold'+str(fold_idx)+'_test'].dropna().astype(int).values
            splits.append((train_idxs, test_idxs))
        return splits

    @classmethod
    def _manual_train_test(cls, data, args):
        if 'train_idx' not in args.keys():
            raise ValueError("{} with 'manual_train_test' strategy must provide a 'train_idx' strategy arg".format(type(cls).__name__))
        if 'test_idx' not in args.keys():
            raise ValueError("{} with 'manual_train_test' strategy must provide a 'test_idx' strategy arg".format(type(cls).__name__))
        return [(list(args['train_idx']), list(args['test_idx']))]
        

    def _validate(self, dc):
        if self._strategy not in self._strategies.keys():
            raise ValueError("Unknown strategy passed to {}; must be one of {}".format(type(self).__name__, list(self._strategies.keys())))
        # Other validation handled by each strategy implementation.
        # TODO - use strategy design pattern to encapsuate validation and execution of each strategy
        return

    def _execute(self, dc):
        Logger.getInst().info("CV fold strategy selected as {}".format(self._strategy))
        strategy = self._strategies[self._strategy]
        Logger.getInst().info("Generating CV Folds")
        X = dc.get_item('data')
        splits = strategy(X, self._strategy_args)
        dc.set_item('cv_splits', splits)
        return dc

GenerateCVFoldsStage._strategies = {
    "random": GenerateCVFoldsStage._random_kfold,
    "random_grouped": GenerateCVFoldsStage._random_grouped_kfold,
    "stratified": GenerateCVFoldsStage._stratified_kfold,
    "stratified_grouped": GenerateCVFoldsStage._stratified_grouped_kfold,
    "load_premade": GenerateCVFoldsStage._load_premade_folds,
    "manual_train_test": GenerateCVFoldsStage._manual_train_test
}



# CLASS DESCRIPTION:
# Treats each subset in the data partition as a test set and remaining data as training set. 
# Uses cross validation to make predictions on each test set. 
# Does not tune hyperparameters.
class CrossValidationStage(StageBase):
    def __init__(self):
        super().__init__()
        self._cv_context = None


    def setCVContext(self, cv_context):
        self._cv_context = cv_context

    def _validate(self, dc):
        if "data" not in dc.get_keys():
            raise ValueError("{} needs a dataframe object named 'data' to be present in the data container".format(type(self).__name__))
        if not isinstance(self._cv_context, SupervisedCVContext):
            raise ValueError("setCVContext only accepts instances of SupervisedCVContext")
        self._cv_context.validate()
        return

    def _execute(self, dc):
        Logger.getInst().info("Starting cross-validation stage")

        cv_splits = dc.get_item("cv_splits")

        cv_results = {}
        data = dc.get_item('data')
        #data = data[cols]  # feature and label cols
        for i in range(len(cv_splits)):
            Logger.getInst().info("Running CV for fold {}".format(i))
            train_idx, test_idx = cv_splits[i]

            p = Pipeline()
            load_stage = DataFrameLoaderStage()
            load_stage.setDataFrame(data, reset_index=True)
            p.addStage(load_stage)

            for stage in self._cv_context.get_preprocessing_stages():
                pre_stage = copy.deepcopy(stage)
                pre_stage.set_fit_transform_data_idx(train_idx)
                pre_stage.set_transform_data_idx(test_idx)
                p.addStage(pre_stage)

            Logger.getInst().info("Training for fold {}".format(i))
            training_stage = ModelTrainingStage(train_idx)
            training_stage.setTrainingContext(self._cv_context.training_context)
            p.addStage(training_stage)

            prediction_stage = ModelPredictionStage(test_idx)
            prediction_context = PredictionContext()
            prediction_context.feature_cols = self._cv_context.training_context.feature_cols
            prediction_stage.setPredictionContext(prediction_context)
            p.addStage(prediction_stage)

            eval_stage = EvaluationStage(test_idx)
            eval_stage.setEvaluationContext(self._cv_context.eval_context)
            p.addStage(eval_stage)

            p.run()
            cv_results['fold'+str(i)] = p.getDC()

        dc.set_item('cv_results', cv_results)
        return dc


class NestedCrossValidationStage(StageBase):
    def __init__(self):
        super().__init__()
        self._nested_cv_context = None

    def set_nested_cv_context(self, nested_cv_context):
        self._nested_cv_context = copy.deepcopy(nested_cv_context)

    def _validate(self, dc):
        if "data" not in dc.get_keys():
            raise ValueError("{} needs a dataframe object named 'data' to be present in the data container".format(type(self).__name__))
        if "cv_splits" not in dc.get_keys():
            raise ValueError("{} needs a set of cross-validation folds named 'cv_splits' to be present in the data container".format(type(self).__name__))
        if not isinstance(self._nested_cv_context, NestedSupervisedCVContext):
            raise ValueError("set_nested_cv_context() requires an instance of NestedSupervisedCVContext")
        self._nested_cv_context.validate()
        return

    def _execute(self, dc):
        Logger.getInst().info("Starting nested cross-validation stage")

        cv_splits = dc.get_item("cv_splits")

        nested_cv_results = {}
        data = dc.get_item('data')
        param_eval_func_name = self._nested_cv_context.tuning_context.get_param_eval_func_name()
        for i in tqdm(range(len(cv_splits))):
            Logger.getInst().info("Running nested CV for fold {}".format(i))
            train_idx, test_idx = cv_splits[i]
            data_train = data.iloc[train_idx, :]

            params_eval_results = []
            params = self._nested_cv_context.tuning_context.get_next_params()
            while params is not None:
                p = Pipeline()
                load_stage = DataFrameLoaderStage()
                load_stage.setDataFrame(data_train, reset_index=True)
                p.addStage(load_stage)

                p.addStage(self._nested_cv_context.cv_folds_stage)

                cv_stage = CrossValidationStage()
                cv_context = SupervisedCVContext()
                cv_context.training_context = copy.copy(self._nested_cv_context.training_context)
                cv_context.training_context.model.set_params(params) # BB - do we need to deepcopy the model?
                cv_context.eval_context = copy.copy(self._nested_cv_context.eval_context)
                cv_context.eval_context.eval_funcs = self._nested_cv_context.tuning_context.param_eval_func
                for stage in self._nested_cv_context.get_preprocessing_stages():
                    cv_context.add_preprocessing_stage(stage)
                cv_stage.setCVContext(cv_context)
                p.addStage(cv_stage)

                p.run()
                cv_results = p.getDC().get_item('cv_results')
                eval_values = [cv_results[fold].get_item('evaluation_results')[param_eval_func_name] for fold in cv_results.keys()]
                params_eval_results.append((params, np.mean(eval_values)))

                params = self._nested_cv_context.tuning_context.get_next_params()
            self._nested_cv_context.tuning_context.reset_next_params()

            params_eval_results.sort(key=lambda x: np.mean(x[1]), reverse=(self._nested_cv_context.tuning_context.param_eval_goal == 'max'))
            best_params = params_eval_results[0][0]

            # Retrain using the best params
            p = Pipeline()
            load_stage = DataFrameLoaderStage()
            load_stage.setDataFrame(data, reset_index=True)
            p.addStage(load_stage)

            for stage in self._nested_cv_context.get_preprocessing_stages():
                pre_stage = copy.deepcopy(stage)
                pre_stage.set_fit_transform_data_idx(train_idx)
                pre_stage.set_transform_data_idx(test_idx)
                p.addStage(pre_stage)

            training_stage = ModelTrainingStage(train_idx)
            training_context = copy.copy(self._nested_cv_context.training_context)
            training_context.model.set_params(best_params)
            training_stage.setTrainingContext(training_context)
            p.addStage(training_stage)

            prediction_stage = ModelPredictionStage(test_idx)
            pred_context = PredictionContext()
            pred_context.feature_cols = training_context.feature_cols
            prediction_stage.setPredictionContext(pred_context)
            p.addStage(prediction_stage)

            eval_stage = EvaluationStage(test_idx)
            eval_stage.setEvaluationContext(self._nested_cv_context.eval_context)
            p.addStage(eval_stage)

            p.run()

            # Save results and the best parameter
            nested_cv_dc = p.getDC()
            nested_cv_dc.set_item('best_param', best_params)
            nested_cv_results['fold'+str(i)] = nested_cv_dc

        dc.set_item('nested_cv_results', nested_cv_results)
        return dc



##################################################################
class ParamSearch(ABC):
    def __init__(self, params):
        self.set_params(params)

    def get_params(self):
        return self._params

    def set_params(self, params):
        self._params = params

    @abstractmethod
    def validate(self):
        if not isinstance(params, dict):
            raise ValueError("{} requires a dictionary of parameters (keys) and values".format(type(self).__name__))

    @abstractmethod
    def reset_next_params(self):
        return

    # BB - prev_param_eval is intended for parameter searches which use previous values
    #      to decide which parameters come next
    @abstractmethod
    def get_next_params(self, prev_param_eval=None):
        return   

    params = property(get_params, set_params)

class GridParamSearch(ParamSearch):
    def set_params(self, params):
        super().set_params(params)
        param_vals = list(itertools.product(*[val for val in self._params.values()]))
        param_keys = list(self._params.keys())
        self._param_grid = [dict(zip(param_keys, param_vals[i])) for i in range(len(param_vals))]
        self._grid_idx = 0

    def validate():
        super().validate()

    def reset_next_params(self):
        self._grid_idx = 0

    def get_next_params(self, prev_param_eval=None):
        if self._grid_idx < len(self._param_grid):
            next_params = self._param_grid[self._grid_idx]
            self._grid_idx += 1
        else:
            next_params = None
        return next_params

class ModelTuningContext(ABC):
    def __init__(self):
        self._model_param_search = None
        self._param_eval_func = None
        self._param_eval_goal = None

    def get_model_param_search(self):
        return self._model_param_search

    def set_model_param_search(self, model_param_search):
        self._model_param_search = model_param_search

    def get_param_eval_func(self):
        return self._param_eval_func

    def get_param_eval_func_name(self):
        if hasattr(self._param_eval_func, '__name__'):
            return self._param_eval_func.__name__
        else:
            return type(self._param_eval_func).__name__

    def set_param_eval_func(self, param_eval_func):
        self._param_eval_func = param_eval_func

    def get_param_eval_goal(self):
        return self._param_eval_goal

    def set_param_eval_goal(self, param_eval_goal):
        self._param_eval_goal = param_eval_goal.lower()

    @abstractmethod
    def validate(self):
        if not isinstance(self._model_param_search, ParamSearch):
            raise ValueError('model_param_search much be set to a subclass of ParamSearch')
        if self.param_eval_func is None:
            raise RuntimeError('param_eval_func must be set before execution')
        if not callable(self.param_eval_func):
            raise ValueError('param_eval_func must be initialized to a callable type')
        if self.param_eval_goal is None:
            raise RuntimeError('param_eval_goal must be initialized to min or max')
        if self.param_eval_goal not in ['min', 'max']:  # TODO: make enum object with min/max types
            raise ValueError('param_eval_goal must be either "min" or "max"')

    def reset_next_params(self):
        self._model_param_search.reset_next_params()

    def get_next_params(self, prev_param_eval=None):
        next_params = self._model_param_search.get_next_params(prev_param_eval)
        if next_params is not None:
            return {'model': next_params}
        else:
            return None

    model_param_search = property(get_model_param_search, set_model_param_search)
    param_eval_func = property(get_param_eval_func, set_param_eval_func)
    param_eval_goal = property(get_param_eval_goal, set_param_eval_goal)

class SklearnModelTuningContext(ModelTuningContext):
    def get_param_eval_func(self):
        return self._param_eval_func

    def set_param_eval_func(self, param_eval_func):
        if isinstance(param_eval_func, str):
            super().set_param_eval_func(get_sklearn_scoring_func(param_eval_func))
        else:
            super().set_param_eval_func(param_eval_func)

    def validate(self):
        super().validate()

    param_eval_func = property(get_param_eval_func, set_param_eval_func)

class TensorFlowModelTuningContext(ModelTuningContext):
    def __init__(self):
        super().__init__()
        self._fit_param_search = None
        self._predict_param_search = None
        self._compile_param_search = None
        self._current_params = None

    def get_param_eval_func(self):
        return self._param_eval_func
    
    def set_param_eval_func(self, param_eval_func):
        if isinstance(param_eval_func, str):
            super().set_param_eval_func(get_tensorflow_loss_func(param_eval_func)) # TODO - support scoring functions
        else:
            super().set_param_eval_func(param_eval_func)

    def get_fit_param_search(self):
        return self._fit_param_search

    def set_fit_param_search(self, fit_param_search):
        self._fit_param_search = fit_param_search

    def get_predict_param_search(self):
        return self._predict_param_search

    def set_predict_param_search(self, predict_param_search):
        self._predict_param_search = predict_param_search

    def get_compile_param_search(self):
        return self._compile_param_search

    def set_compile_param_search(self, compile_param_search):
        self._compile_param_search = compile_param_search

    def reset_next_params(self):
        self._model_param_search.reset_next_params()
        self._fit_param_search.reset_next_params()
        self._predict_param_search.reset_next_params()
        self._compile_param_search.reset_next_params()

    def get_next_params(self, prev_param_eval=None):
        if self._current_params is None:
            self._current_params = {}
            if self._model_param_search is not None:
                self._current_params['model'] = self._model_param_search.get_next_params(prev_param_eval)
            if self._fit_param_search is not None:
                self._current_params['fit'] = self._fit_param_search.get_next_params(prev_param_eval)
            if self._predict_param_search is not None:
                self._current_params['predict'] = self._predict_param_search.get_next_params(prev_param_eval)
            if self._compile_param_search is not None:
                self._current_params['compile'] = self._compile_param_search.get_next_params(prev_param_eval)
        else:
            # Cycle 'model' params first
            need_next_model_params = False
            if self._model_param_search is not None:
                next_model_params = self._model_param_search.get_next_params(prev_param_eval)
                need_next_model_params = True
            else:
                next_model_params = None

            if next_model_params is None:
                if need_next_model_params:
                    self._model_param_search.reset_next_params()
                    self._current_params['model'] = self._model_param_search.get_next_params(prev_param_eval)

                # Cycle 'fit' params second
                need_next_fit_params = False
                if self._fit_param_search is not None:
                    next_fit_params = self._fit_param_search.get_next_params(prev_param_eval)
                    need_next_fit_params = True
                else:
                    next_fit_params = None

                if next_fit_params is None:
                    if need_next_fit_params:
                        self._fit_param_search.reset_next_params()
                        self._current_params['fit'] = self._model_fit_search.get_next_params(prev_param_eval)

                    # Cycle 'predict' params third
                    need_next_predict_params = False
                    if self._predict_param_search is not None:
                        next_predict_params = self._predict_param_search.get_next_params(prev_param_eval)
                        need_next_predict_params = True
                    else:
                        next_predict_params = None

                    if next_predict_params is None:
                        if need_next_predict_params:
                            self._predict_param_search.reset_next_params()
                            self._current_params['predict'] = self._model_predict_search.get_next_params(prev_param_eval)
                            
                        # Cycle 'compile' params fourth
                        if self._compile_param_search is not None:
                            next_compile_params = self._compile_param_search.get_next_params(prev_param_eval)
                            if next_compile_params is not None:
                                self._current_params['compile'] = next_compile_params
                            else:
                                self._current_params = None
                        else:
                            self._current_params = None
                    else:
                        self._current_params['predict'] = next_predict_params
                else:
                    self._current_params['fit'] = next_fit_params
            else:
                self._current_params['model'] = next_model_params

        return self._current_params

    def validate(self):
        super().validate()
        if self._fit_param_search is not None and not isinstance(self._fit_param_search, ParamSearch):
            raise ValueError('fit_param_search much be set to a subclass of ParamSearch')
        if self._predict_param_search is not None and not isinstance(self._predict_param_search, ParamSearch):
            raise ValueError('predict_param_search much be set to a subclass of ParamSearch')
        if self._compile_param_search is not None and not isinstance(self._compile_param_search, ParamSearch):
            raise ValueError('compile_param_search much be set to a subclass of ParamSearch')

    param_eval_func = property(get_param_eval_func, set_param_eval_func)
    fit_param_search = property(get_fit_param_search, set_fit_param_search)
    predict_param_search = property(get_predict_param_search, set_predict_param_search)
    compile_param_search = property(get_compile_param_search, set_compile_param_search)



class PyTorchModelTuningContext(ModelTuningContext):
    def __init__(self):
        super().__init__()
        self._fit_param_search = None
        self._criterion_param_search = None
        self._optimizer_param_search = None
        self._current_params = None

    def get_param_eval_func(self):
        return self._param_eval_func

    def set_param_eval_func(self, param_eval_func):
        if isinstance(param_eval_func, str):
            super().set_param_eval_func(get_torch_loss_func(param_eval_func)) # TODO - support scoring functions
        else:
            super().set_param_eval_func(param_eval_func)

    def get_fit_param_search(self):
        return self._fit_param_search

    def set_fit_param_search(self, fit_param_search):
        self._fit_param_search = fit_param_search

    def get_criterion_param_search(self):
        return self._criterion_param_search

    def set_criterion_param_search(self, criterion_param_search):
        self._criterion_param_search = criterion_param_search

    def get_optimizer_param_search(self):
        return self._optimizer_param_search

    def set_optimizer_param_search(self, optimizer_param_search):
        self._optimizer_param_search = optimizer_param_search

    def reset_next_params(self):
        self._model_param_search.reset_next_params()
        self._fit_param_search.reset_next_params()
        self._criterion_param_search.reset_next_params()
        self._optimizer_param_search.reset_next_params()

    def get_next_params(self, prev_param_eval=None):
        if self._current_params is None:
            self._current_params = {}
            if self._model_param_search is not None:
                self._current_params['model'] = self._model_param_search.get_next_params(prev_param_eval)
            if self._fit_param_search is not None:
                self._current_params['fit'] = self._fit_param_search.get_next_params(prev_param_eval)
            if self._criterion_param_search is not None:
                self._current_params['criterion'] = self._criterion_param_search.get_next_params(prev_param_eval)
            if self._optimizer_param_search is not None:
                self._current_params['optimizer'] = self._optimizer_param_search.get_next_params(prev_param_eval)
        else:
            # Cycle 'model' params first
            need_next_model_params = False
            if self._model_param_search is not None:
                next_model_params = self._model_param_search.get_next_params(prev_param_eval)
                need_next_model_params = True
            else:
                next_model_params = None

            if next_model_params is None:
                if need_next_model_params:
                    self._model_param_search.reset_next_params()
                    self._current_params['model'] = self._model_param_search.get_next_params(prev_param_eval)

                # Cycle 'fit' params second
                need_next_fit_params = False
                if self._fit_param_search is not None:
                    next_fit_params = self._fit_param_search.get_next_params(prev_param_eval)
                    need_next_fit_params = True
                else:
                    next_fit_params = None

                if next_fit_params is None:
                    if need_next_fit_params:
                        self._fit_param_search.reset_next_params()
                        self._current_params['fit'] = self._fit_param_search.get_next_params(prev_param_eval)

                    # Cycle 'criterion' params third
                    need_next_criterion_params = False
                    if self._criterion_param_search is not None:
                        next_criterion_params = self._criterion_param_search.get_next_params(prev_param_eval)
                        need_next_criterion_params = True
                    else:
                        next_criterion_params = None

                    if next_criterion_params is None:
                        if need_next_criterion_params:
                            self._criterion_param_search.reset_next_params()
                            self._current_params['criterion'] = self._criterion_param_search.get_next_params(prev_param_eval)
                            
                        # Cycle 'optimizer' params fourth
                        if self._optimizer_param_search is not None:
                            next_optimizer_params = self._optimizer_param_search.get_next_params(prev_param_eval)
                            if next_optimizer_params is not None:
                                self._current_params['optimizer'] = next_optimizer_params
                            else:
                                self._current_params = None
                        else:
                            self._current_params = None
                    else:
                        self._current_params['criterion'] = next_criterion_params
                else:
                    self._current_params['fit'] = next_fit_params
            else:
                self._current_params['model'] = next_model_params

        return self._current_params

    def validate(self):
        super().validate()
        if self._fit_param_search is not None and not isinstance(self._fit_param_search, ParamSearch):
            raise ValueError('fit_param_search much be set to a subclass of ParamSearch')
        if self._criterion_param_search is not None and not isinstance(self._criterion_param_search, ParamSearch):
            raise ValueError('criterion_param_search much be set to a subclass of ParamSearch')
        if self._optimizer_param_search is not None and not isinstance(self._optimizer_param_search, ParamSearch):
            raise ValueError('optimizer_param_search much be set to a subclass of ParamSearch')

    param_eval_func = property(get_param_eval_func, set_param_eval_func)
    fit_param_search = property(get_fit_param_search, set_fit_param_search)
    criterion_param_search = property(get_criterion_param_search, set_criterion_param_search)
    optimizer_param_search = property(get_optimizer_param_search, set_optimizer_param_search)


class SupervisedCVContext():
    def __init__(self):
        self._training_context = None
        self._eval_context = None
        self._preprocessing_stages = []

    def get_training_context(self):
        return self._training_context

    def set_training_context(self, training_context):
        self._training_context = training_context

    def get_eval_context(self):
        return self._eval_context

    def set_eval_context(self, eval_context):
        self._eval_context = eval_context

    def get_preprocessing_stages(self):
        return self._preprocessing_stages

    def add_preprocessing_stage(self, stage):
        self._preprocessing_stages.append(stage)

    def validate(self):
        for stage in self._preprocessing_stages:
            if not isinstance(stage, PreprocessingStageBase):
                raise ValueError("add_preprocessing_stage() only accepts instances of PreprocessingStageBase")
        if not isinstance(self._training_context, SupervisedTrainingContext):
            raise ValueError("set_training_context requires an instance of SupervisedTrainingContext")
        if not isinstance(self._eval_context, SupervisedEvaluationContext):
            raise ValueError("set_eval_context requires an instance of SupervisedEvaluationContext")
        self._training_context.validate()
        self._eval_context.validate()

    training_context = property(get_training_context, set_training_context)
    eval_context = property(get_eval_context, set_eval_context)


class NestedSupervisedCVContext(SupervisedCVContext):
    def __init__(self):
        super().__init__()
        self._tuning_context = None
        self._cv_folds_stage = None

    def get_tuning_context(self):
        return self._tuning_context

    def set_tuning_context(self, tuning_context):
        self._tuning_context = tuning_context

    def get_cv_folds_stage(self):
        return self._cv_folds_stage

    def set_cv_folds_stage(self, cv_folds_stage):
        self._cv_folds_stage = cv_folds_stage

    def validate(self):
        if not isinstance(self._cv_folds_stage, GenerateCVFoldsStage):
            raise ValueError("set_valiation_cv_folds_stage() requires an instance of GenerateCVFoldsStage")
        if not isinstance(self._tuning_context, ModelTuningContext):
            raise ValueError("set_tuning_context requires an instance of ModelTuningContext")
        self._tuning_context.validate()

    tuning_context = property(get_tuning_context, set_tuning_context)
    cv_folds_stage = property(get_cv_folds_stage, set_cv_folds_stage)
