import os
import copy
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from collections.abc import Iterable, Mapping
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from .stage_base import StageBase
from .pipeline import Pipeline
from ..models.model import ModelBase, PyTorchModel, SklearnModel, TensorFlowModel
from .load_data import ObjectDataLoaderStage
from .preprocessing import PreprocessingStageBase
from .training_stage import ModelTrainingStage, SupervisedTrainingContext
from .evaluation_stage import EvaluationStage, SupervisedEvaluationContext
from .prediction_stage import ModelPredictionStage
from ..utils.utils import get_sklearn_scoring_func, get_tensorflow_loss_func, get_any_scoring_func
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
            y = pd.qcut(data['stratify_on'].argsort(kind='stable'), q=args['percentile_bins'], labels=range(split)).tolist()
        else:
            y = np.digitize(y, bins=args['bin_edges'])
        return skf.split(data, y)

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
        

    def _validate(self, dc):
        if self._strategy not in self._strategies.keys():
            raise ValueError("Unknown strategy passed to {}; must be one of {}".format(GenerateCVFoldsStage.__name__, GenerateCVFoldsStage._strategies))
        # Other validation handled by each strategy implementation.
        # TODO - use strategy design pattern to encapsuate validation and execution of each strategy
        return

    def _execute(self, dc):
        self.logInfo("CV fold strategy selected as {}".format(self._strategy))
        strategy = self._strategies[self._strategy]
        self.logInfo("Generating CV Folds")
        X = dc.get_item('data')
        splits = strategy(X, self._strategy_args)
        dc.set_item('cv_splits', splits)
        return dc

GenerateCVFoldsStage._strategies = {
    "random": GenerateCVFoldsStage._random_kfold,
    "random_grouped": GenerateCVFoldsStage._random_grouped_kfold,
    "stratified": GenerateCVFoldsStage._stratified_kfold,
    "stratified_grouped": GenerateCVFoldsStage._stratified_grouped_kfold,
    "load_premade": GenerateCVFoldsStage._load_premade_folds
}



# CLASS DESCRIPTION:
# Treats each subset in the data partition as a test set and remaining data as training set. 
# Uses cross validation to make predictions on each test set. 
# Does not tune hyperparameters.
class CrossValidationStage(StageBase):
    def __init__(self):
        super().__init__()
        self._preprocessing_stages = []
        self._cv_context = None
        self._evaluation_context = None
        self.setLoggingPrefix('CrossValidationStage: ')

    def addPreprocessingStage(self, stage):
        self._preprocessing_stages.append(stage)

    def setCVContext(self, cv_context):
        self._cv_context = copy.deepcopy(cv_context)

    def setEvaluationContext(self, eval_context):
        self._evaluation_context = eval_context

    def _validate(self, dc):
        for stage in self._preprocessing_stages:
            if not isinstance(stage, PreprocessingStageBase):
                raise ValueError("addPreprocessingStage only accepts instances of PreprocessingStageBase")
        if "data" not in dc.get_keys():
            raise ValueError("{} needs a dataframe object named 'data' to be present in the data container".format(type(self).__name__))
        if not isinstance(self._cv_context, SupervisedCVContext):
            raise ValueError("setCVContext only accepts instances of SupervisedCVContext")
        if not isinstance(self._evaluation_context, SupervisedEvaluationContext):
            raise ValueError("setEvaluationContext only accepts instances of SupervisedEvaluationContext")
        # TODO - check that param grid has a single param per key?
        self._cv_context.validate()
        self._evaluation_context.validate()
        return

    def _execute(self, dc):
        self.logInfo("Starting cross-validation stage")

        #cols = [x for x in self._cv_context.feature_cols]
        #if isinstance(self._cv_context.y_label, Iterable):
        #    cols = cols + [x for x in self._cv_context.y_label]
        #else:
        #    cols.append(self._cv_context.y_label)

        cv_splits = dc.get_item("cv_splits")

        # unpack cv_context. Should be one item per key
        # QUESTION: How does this work? Setting the param_grid to one value  
        # param_grid = self._cv_context.get_param_grid()
        # for d in param_grid.values():
        #     for k in d.keys():
        #         if not isinstance(d[k], str) and isinstance(d[k], Iterable):
        #             item = next(d[k], None)
        #             d[k] = item
        # self._cv_context.set_param_grid(param_grid)

        cv_results = {}
        data = dc.get_item('data')
        #data = data[cols]  # feature and label cols
        for i in range(len(cv_splits)):
            self.logInfo("Running CV for fold {}".format(i))
            train_idx, test_idx = cv_splits[i]

            p = Pipeline()
            load_stage = ObjectDataLoaderStage()
            load_stage.setDataObject(data, reset_index=True)
            p.addStage(load_stage)

            for stage in self._preprocessing_stages:
                stage.set_fit_transform_data_idx(train_idx)
                stage.set_transform_data_idx(test_idx)
                p.addStage(stage)

            training_stage = ModelTrainingStage(train_idx)
            training_stage.setTrainingContext(self._cv_context)
            p.addStage(training_stage)

            prediction_stage = ModelPredictionStage(test_idx)
            prediction_stage.setPredictionContext(self._cv_context)
            p.addStage(prediction_stage)

            eval_stage = EvaluationStage(test_idx)
            eval_stage.setEvaluationContext(self._evaluation_context)
            p.addStage(eval_stage)

            p.run()
            cv_results['fold'+str(i)] = p.getDC()

        dc.set_item('cv_results', cv_results)
        return dc


class NestedCrossValidationStage(StageBase):
    def __init__(self):
        super().__init__()
        self._preprocessing_stages = []
        self._nested_cv_context = None
        self._cv_folds_stage = None
        self._evaluation_context = None
        self.setLoggingPrefix('NestedCrossValidationStage: ')

    def addPreprocessingStage(self, stage):
        self._preprocessing_stages.append(stage)

    def setCVFoldsStage(self, stage):
        self._cv_folds_stage = stage

    def setCVContext(self, nested_cv_context):
        self._nested_cv_context = copy.deepcopy(nested_cv_context)

    def setEvaluationContext(self, eval_context):
        self._evaluation_context = eval_context 

    # def _createIterableParameterGrid(self, param_grid):
    #     params_flat = {k:v for d in param_grid.values() for k,v in d.items()}
    #     param_vals = list(itertools.product(*[v for v in params_flat.values()]))
    #     param_keys = [k for k in params_flat.keys()]
    #     param_grid_list = [dict(zip(param_keys, param_vals[i])) for i in range(len(param_vals))]
    #     # Remake each parameter dictionary into the correct structure
    #     res = []
    #     for param_dict in param_grid_list:
    #         new_param_dict = {}
    #         for param_type, d in param_grid.items():
    #             new_param_dict[param_type] = {k:v for k,v in param_dict.items() if k in d}
    #         res.append(new_param_dict)
    #     return res

    def _validate(self, dc):
        for stage in self._preprocessing_stages:
            if not isinstance(stage, PreprocessingStageBase):
                raise ValueError("addPreprocessingStage only accepts instances of PreprocessingStageBase")
        if "data" not in dc.get_keys():
            raise ValueError("{} needs a dataframe object named 'data' to be present in the data container".format(type(self).__name__))
        if "cv_splits" not in dc.get_keys():
            raise ValueError("{} needs a set of cross-validation folds named 'cv_splits' to be present in the data container".format(type(self).__name__))
        if not isinstance(self._cv_folds_stage, GenerateCVFoldsStage):
            raise ValueError("setCVFoldsStage requires an instance of GenerateCVFoldsStage")
        if not isinstance(self._nested_cv_context, NestedSupervisedCVContext):
            raise ValueError("setCVContext requires an instance of SupervisedCVContext")
        if not isinstance(self._evaluation_context, SupervisedEvaluationContext):
            raise ValueError("setEvaluationContext requires an instance of SupervisedEvaluationContext")
        self._nested_cv_context.validate()
        self._evaluation_context.validate()
        return

    def _execute(self, dc):
        self.logInfo("Starting nested cross-validation stage")

        #cols = [x for x in self._nested_cv_context.feature_cols]
        #if isinstance(self._nested_cv_context.y_label, Iterable):
        #    cols = cols + [x for x in self._nested_cv_context.y_label]
        #else:
        #    cols.append(self._nested_cv_context.y_label)

        cv_splits = dc.get_item("cv_splits")

        # unpack cv_context
        param_grid = self._nested_cv_context.create_iterable_param_grid()
        param_eval_goal = self._nested_cv_context.param_eval_goal
        param_eval_func = self._nested_cv_context.param_eval_func

        nested_cv_results = {}
        data = dc.get_item('data')
        #data = data[cols] # x and y cols
        for i in tqdm(range(len(cv_splits))):
            self.logInfo("Running nested CV for fold {}".format(i))
            train_idx, test_idx = cv_splits[i]
            data_train = data.iloc[train_idx, :]

            param_grid_eval_results = []
            for point in param_grid:
                p = Pipeline()
                load_stage = ObjectDataLoaderStage()
                load_stage.setDataObject(data_train, reset_index=True)
                p.addStage(load_stage)

                p.addStage(self._cv_folds_stage)

                cv_stage = CrossValidationStage()
                cv_context = self._nested_cv_context.create_cv_context(point)
                cv_stage.setCVContext(cv_context)
                for stage in self._preprocessing_stages:
                    cv_stage.addPreprocessingStage(stage)

                eval_context = SupervisedEvaluationContext()
                eval_context.y_label = self._evaluation_context.y_label
                eval_context.eval_funcs = param_eval_func
                param_eval_func_name = eval_context.get_eval_func_names()[0]
                cv_stage.setEvaluationContext(eval_context)
                p.addStage(cv_stage)

                p.run()
                cv_results = p.getDC().get_item('cv_results')
                eval_values = [cv_results[fold].get_item('evaluation_results')[param_eval_func_name] for fold in cv_results.keys()]
                param_grid_eval_results.append((point, np.mean(eval_values)))

            param_grid_eval_results.sort(key=lambda x: np.mean(x[1]), reverse=(param_eval_goal == 'max'))
            best_param_point = param_grid_eval_results[0][0]

            # Retrain using the best param point
            p = Pipeline()
            load_stage = ObjectDataLoaderStage()
            load_stage.setDataObject(data, reset_index=True)
            p.addStage(load_stage)

            for stage in self._preprocessing_stages:
                stage.set_fit_transform_data_idx(train_idx)
                stage.set_transform_data_idx(test_idx)
                p.addStage(stage)

            training_stage = ModelTrainingStage(train_idx)
            cv_context = self._nested_cv_context.create_cv_context(best_param_point)
            training_stage.setTrainingContext(cv_context)
            p.addStage(training_stage)

            prediction_stage = ModelPredictionStage(test_idx)
            prediction_stage.setPredictionContext(cv_context)
            p.addStage(prediction_stage)

            eval_stage = EvaluationStage(test_idx)
            eval_stage.setEvaluationContext(self._evaluation_context)
            p.addStage(eval_stage)

            p.run()

            # Save results and the best parameter
            nested_cv_dc = p.getDC()
            nested_cv_dc.set_item('best_param', best_param_point)
            nested_cv_results['fold'+str(i)] = nested_cv_dc

        dc.set_item('nested_cv_results', nested_cv_results)
        return dc



##################################################################
class SupervisedCVContext(SupervisedTrainingContext):
    def __init__(self):
        super().__init__()
        self._param_eval_func = None # TODO - this should live on the nested CV context
        self._param_eval_goal = None
        # self._param_grid = None
        
    # Might be best to just set these on the model when this is called. I don't see any reason otherwise
    # def get_params(self):
    #     return super().get_params()

    # def set_params(self, params):
    #     super().set_params(params)
    #     return        

    def get_param_eval_func_name(self):
        return self._param_eval_func.__name__

    def get_param_eval_func(self):
        return self._param_eval_func

    def set_param_eval_func(self, param_eval_func):
        self._param_eval_func = param_eval_func
        return

    def get_param_eval_goal(self):
        return self._param_eval_goal

    def set_param_eval_goal(self, param_eval_goal):
        self._param_eval_goal = param_eval_goal.lower()
        return

    def validate(self):
        super().validate()
        # if self.param_eval_func is None:
        #     raise RuntimeError('param_eval_func must be set before execution')
        # if not callable(self.param_eval_func):
        #     raise ValueError('param_eval_func must be initialized to a callable type')
        # if self.param_eval_goal is None:
        #     raise RuntimeError('param_eval_goal must be initialized to min or max')
        # if self.param_eval_goal not in ['min', 'max']:  # TODO: make enum object with min/max types
        #     raise ValueError('param_eval_goal must be either "min" or "max"')
        return

    # param_grid = property(get_param_grid, set_param_grid)
    param_eval_func = property(get_param_eval_func, set_param_eval_func)
    param_eval_goal = property(get_param_eval_goal, set_param_eval_goal)

class NestedSupervisedCVContext(SupervisedCVContext):
    def __init__(self):
        super().__init__()
        self._param_grid = None # BB TODO - can we always use this as the model params and name it as such? Reuse this in the subclasses instead of adding new model_params!

    # TODO: These should be used by the user and implemented by backend specific classes
    # def get_params(self):
    #     return super().get_params()

    # def set_params(self, params):
    #     # Convert any non iterables to lists
    #     for d in params.values():
    #         for k,v in d.items():
    #             if not isinstance(v,list):
    #                 d[k] = [v]
    #     super().set_params(params)
    #     return

    # I don't think this is the correct way to do this        
    def create_cv_context(self, params):
        cv_context = SupervisedCVContext()
        # I'm not sure if you should be able to set a model property on nested cv context
        model = copy.deepcopy(self.model)
        model.set_params(params)
        cv_context.model = model
        cv_context.feature_cols = self.feature_cols
        cv_context.y_label = self.y_label
        cv_context.param_eval_func = self.param_eval_func
        cv_context.param_eval_goal = self.param_eval_goal
        return cv_context
        
    def get_param_grid(self):
        return self._param_grid

    def set_param_grid(self, param_grid):
        self._param_grid = param_grid

    def create_iterable_param_grid(self):
        param_grid = self.get_param_grid()
        params_flat = {k:v for d in param_grid.values() for k,v in d.items()}
        param_vals = list(itertools.product(*[v for v in params_flat.values()]))
        param_keys = [k for k in params_flat.keys()]
        param_grid_list = [dict(zip(param_keys, param_vals[i])) for i in range(len(param_vals))]
        # Remake each parameter dictionary into the correct structure
        res = []
        for param_dict in param_grid_list:
            new_param_dict = {}
            for param_type, d in param_grid.items():
                new_param_dict[param_type] = {k:v for k,v in param_dict.items() if k in d}
            res.append(new_param_dict)
        return res

    def validate(self):
        super().validate()
        if self.param_eval_func is None:
            raise RuntimeError('param_eval_func must be set before execution')
        if not callable(self.param_eval_func):
            raise ValueError('param_eval_func must be initialized to a callable type')
        if self.param_eval_goal is None:
            raise RuntimeError('param_eval_goal must be initialized to min or max')
        if self.param_eval_goal not in ['min', 'max']:  # TODO: make enum object with min/max types
            raise ValueError('param_eval_goal must be either "min" or "max"')
        return

    param_grid = property(get_param_grid, set_param_grid)


class SklearnNestedSupervisedCVContext(NestedSupervisedCVContext):
    def __init__(self):
        super().__init__()
        self._model_params = None

    def get_param_grid(self):
        param_grid = {}
        model_key, model_dict = self.get_model_params()
        if model_dict:
            param_grid[model_key] = model_dict
        return param_grid      

    def get_model_params(self, key="model"):
        if key:
            return key, self._model_params
        else:
            return self._model_params

    def set_model_params(self, params):
        self._model_params = params

    def get_params(self):
        return self.param_grid

    def set_params(self, param_dict):
        self.param_grid = param_dict
        return

    def get_param_eval_func(self):
        return self._param_eval_func

    def set_param_eval_func(self, param_eval_func):
        if isinstance(param_eval_func, str):
            self.param_eval_func = get_sklearn_scoring_func(param_eval_func)
        else:
            super().set_param_eval_func(param_eval_func)

    def validate(self):
        super().validate()
        if not isinstance(self.model, SklearnModel):
            raise TypeError("{} must take SklearnModel type as model arg".format(type(self).__name__))
        return True

    param_eval_func = property(get_param_eval_func, set_param_eval_func)
    model_params = property(get_model_params, set_model_params)


class TensorFlowNestedSupervisedCVContext(NestedSupervisedCVContext):
    tf_optimizers = [
        'adadelta',
        'adagrad',
        'adam',
        'adamax',
        'nadam',
        'rmsprop',
        'sgd',
        'ftrl',
        'lossscaleoptimizer',
        'lossscaleoptimizerv1'
    ]

    def __init__(self):
        super().__init__()
        self._model_params = None
        self._fit_params = None
        self._predict_params = None
        self._compile_params = None

    def get_param_eval_func(self):
        return self._param_eval_func

    def set_param_eval_func(self, param_eval_func):
        if isinstance(param_eval_func, str):
            self.param_eval_func = get_tensorflow_loss_func(param_eval_func) # TODO - support metric functions?
        else:
            super().set_param_eval_func(param_eval_func)
        return

    def get_model_params(self, key="model"):
        if key:
            return key, self._model_params
        else:
            return self._model_params
    
    def set_model_params(self, model_params):
        self._model_params = model_params

    def get_fit_params(self, key="fit"):
        if key:
            return key, self._fit_params
        else:
            return self._fit_params
            
    def set_fit_params(self, fit_params):
        self._fit_params = fit_params

    def get_predict_params(self, key="predict"):
        if key:
            return key, self._predict_params
        else:
            return self._predict_params
    
    def set_predict_params(self, predict_params):
        self._predict_params = predict_params
    
    def get_compile_params(self, key="compile"):
        if key:
            return key, self._compile_params
        else:
            return self._compile_params

    def set_compile_params(self, compile_params):
        self._compile_params = compile_params

    def get_param_grid(self):
        param_grid = {}
        model_key, model_dict = self.get_model_params()
        if model_dict:
            param_grid[model_key] = model_dict
        fit_key, fit_dict = self.get_fit_params()
        if fit_dict:
            param_grid[fit_key] = fit_dict
        predict_key, predict_dict = self.get_predict_params()
        if predict_dict:
            param_grid[predict_key] = predict_dict
        compile_key, compile_dict = self.get_compile_params()
        if compile_dict:
            param_grid[compile_key] = compile_dict
        return param_grid
    # def get_optimizer(self):
    #     return self._optimizer

    # def set_optimizer(self, optimizer):
    #     self._optimizer = optimizer.lower()
    #     return

    # def validate(self):
    #     super().validate()
    #     if not isinstance(self.model, TensorFlowModel):
    #         raise TypeError("{} must take TensorFlowModel type as model arg".format(type(self).__name__))
    #     if not isinstance(self.optimizer, str):
    #         raise TypeError("{} optimizer must be string type arg".format(type(self).__name__))
    #     if not self._optimizer in self.tf_optimizers:
    #         raise ValueError("optimizer '{}' not recognized and may not be supported by TensorFlow".format(self._optimizer))
    #     return

    param_eval_func = property(get_param_eval_func, set_param_eval_func)
    model_params = property(get_model_params, set_model_params)
    predict_params = property(get_predict_params, set_predict_params)
    fit_params = property(get_fit_params, set_fit_params)
    compile_params = property(get_compile_params, set_compile_params)
    #optimizer = property(get_optimizer, set_optimizer)

class PyTorchNestedSupervisedCVContext(NestedSupervisedCVContext):
    def __init__(self):
        super().__init__()
        self._model_params = None
        self._optimizer_params = None
        self._criterion_params = None
        self._fit_params = None

    def get_param_grid(self):
        param_grid = {}
        model_key, model_dict = self.get_model_params()
        if model_dict:
            param_grid[model_key] = model_dict
        fit_key, fit_dict = self.get_fit_params()
        if fit_dict:
            param_grid[fit_key] = fit_dict
        optimizer_key, optimizer_dict = self.get_optimizer_params()
        if optimizer_dict:
            param_grid[optimizer_key] = optimizer_dict
        criterion_key, criterion_dict = self.get_criterion_params()
        if criterion_dict:
            param_grid[criterion_key] = criterion_dict
        return param_grid
    
    def get_model_params(self, key="model"):
        if key:
            return key, self._model_params
        else:
            return self._model_params
    
    def set_model_params(self, model_params):
        self._model_params = model_params

    def get_fit_params(self, key="fit"):
        if key:
            return key, self._fit_params
        else:
            return self._fit_params
            
    def set_fit_params(self, fit_params):
        self._fit_params = fit_params

    def get_optimizer_params(self, key="optimizer"):
        if key:
            return key, self._optimizer_params
        else:
            return self._optimizer_params
    
    def set_optimizer_params(self, optimizer_params):
        self._optimizer_params = optimizer_params

    def get_criterion_params(self, key="criterion"):
        if key:
            return key, self._criterion_params
        else:
            return self._criterion_params

    def set_criterion_params(self, criterion_params):
        self._criterion_params = criterion_params

    def get_param_eval_func(self):
        return self._param_eval_func

    def set_param_eval_func(self, param_eval_func):
        if isinstance(param_eval_func, str):
            self.param_eval_func = get_sklearn_scoring_func(param_eval_func)
        else:
            super().set_param_eval_func(param_eval_func)

    def validate(self):
        super().validate()
        if not isinstance(self.model, PyTorchModel):
            raise TypeError("{} must take PyTorchModel type as model arg".format(type(self).__name__))
        return True

    param_eval_func = property(get_param_eval_func, set_param_eval_func)
    model_params = property(get_model_params, set_model_params)
    fit_params = property(get_fit_params, set_fit_params)
    optimizer_params = property(get_optimizer_params, set_optimizer_params)
    criterion_params = property(get_criterion_params, set_criterion_params)

