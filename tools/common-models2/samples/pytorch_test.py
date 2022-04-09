import os 
import numpy as np
import pandas as pd
from torch import nn
from sklearn.datasets import load_iris

from commonmodels2.stages.pipeline import Pipeline
from commonmodels2.models.model import PyTorchModel
from commonmodels2.stages.load_data import DataFrameLoaderStage
from commonmodels2.stages.preprocessing import ImputerPreprocessingStage, FeatureScalerPreprocessingStage
from commonmodels2.stages.cross_validation import GridParamSearch, GenerateCVFoldsStage, NestedCrossValidationStage, NestedSupervisedCVContext, PyTorchModelTuningContext
from commonmodels2.stages.training_stage import SupervisedTrainingContext
from commonmodels2.stages.evaluation_stage import SupervisedEvaluationContext
from commonmodels2.utils.utils import get_tensorflow_metric_func

def pytorch_model_func(params):
    num_classes = 3
    model = nn.Sequential()
    for layer_idx in range(params['num_layers']):
        model.add_module('linear'+str(layer_idx), nn.LazyLinear(params['layer_width']))
        model.add_module('activation'+str(layer_idx), params['activation_func']())

    model.add_module('linear'+str(params['num_layers']), nn.LazyLinear(num_classes))
    return model

def RunML():
    iris_data = load_iris()
    iris_features_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names).astype(np.float32)
    iris_labels_df = pd.DataFrame(data=iris_data.target, columns=['species']).astype(np.int64)
    iris_df = pd.concat((iris_features_df, iris_labels_df), axis=1)

    p = Pipeline()

    s0 = DataFrameLoaderStage()
    s0.setDataFrame(iris_df)
    p.addStage(s0)

    s1 = GenerateCVFoldsStage(strategy='stratified',
                              strategy_args={'num_folds': 3,
                                             'stratify_on': 'species',
                                             'bin_edges': np.unique(iris_labels_df),
                                             'seed': 42})
    p.addStage(s1)

    s2 = NestedCrossValidationStage()
    ncv_context = NestedSupervisedCVContext()
    ncv_context.add_preprocessing_stage(ImputerPreprocessingStage(iris_features_df.columns, 'mean'))
    ncv_context.add_preprocessing_stage(FeatureScalerPreprocessingStage(iris_features_df.columns, 'min-max'))
    training_context = SupervisedTrainingContext()
    pym = PyTorchModel()
    pym.set_model_create_func(pytorch_model_func)
    training_context.model = pym
    training_context.feature_cols = iris_features_df.columns
    training_context.label_cols = iris_labels_df.columns
    ncv_context.training_context = training_context

    tuning_context = PyTorchModelTuningContext()
    tuning_context.model_param_search = GridParamSearch({"num_layers": [1,2], "layer_width": [8,16], "activation_func": [nn.ReLU]})
    tuning_context.fit_param_search = GridParamSearch({"lr": [1e-2], "batch_size": [32], "train_split": [None], "max_epochs": [500]})
    tuning_context.criterion_param_search = GridParamSearch({"criterion": ["categorical_crossentropy"]})
    tuning_context.optimizer_param_search = GridParamSearch({"optimizer": ["adam", "sgd"]})
    tuning_context.param_eval_func = get_tensorflow_metric_func('SparseCategoricalAccuracy')
    tuning_context.param_eval_goal = 'max'
    ncv_context.tuning_context = tuning_context

    eval_context = SupervisedEvaluationContext()
    eval_context.label_cols = iris_labels_df.columns
    eval_context.eval_funcs = [get_tensorflow_metric_func('SparseCategoricalAccuracy')]
    ncv_context.eval_context = eval_context

    cv_folds_stage = GenerateCVFoldsStage(strategy='stratified',
                                          strategy_args={'num_folds': 3,
                                                         'stratify_on': 'species',
                                                         'bin_edges': np.unique(iris_labels_df),
                                                         'seed': 42})
    ncv_context.cv_folds_stage = cv_folds_stage
    s2.set_nested_cv_context(ncv_context)

    p.addStage(s2)

    p.run()

    p.getDC().save('pytorch_test_results')

if __name__ == '__main__':
    RunML()
