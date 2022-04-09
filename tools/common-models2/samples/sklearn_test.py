import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from commonmodels2.stages.pipeline import Pipeline

from commonmodels2.models.model import SklearnModel
from commonmodels2.stages.load_data import DataFrameLoaderStage
from commonmodels2.stages.cross_validation import GridParamSearch, GenerateCVFoldsStage, NestedCrossValidationStage, NestedSupervisedCVContext, SklearnModelTuningContext
from commonmodels2.stages.training_stage import SupervisedTrainingContext
from commonmodels2.stages.evaluation_stage import SupervisedEvaluationContext

def my_sklearn_model_func(params):
    model = DecisionTreeClassifier()
    model.set_params(**params)
    return model

def RunML():
    iris_data = load_iris()
    iris_features_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    iris_labels_df = pd.DataFrame(data=iris_data.target, columns=['species'])
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
    training_context = SupervisedTrainingContext()
    skm = SklearnModel()
    skm.set_model_create_func(my_sklearn_model_func)
    training_context.model = skm
    training_context.feature_cols = iris_features_df.columns
    training_context.label_cols = iris_labels_df.columns
    ncv_context.training_context = training_context

    tuning_context = SklearnModelTuningContext()
    tuning_context.model_param_search = GridParamSearch({'max_depth': [1,2,3], 'random_state': [42]})
    tuning_context.param_eval_func = 'accuracy'
    tuning_context.param_eval_goal = 'max'
    ncv_context.tuning_context = tuning_context

    eval_context = SupervisedEvaluationContext()
    eval_context.label_cols = iris_labels_df.columns
    eval_context.eval_funcs = ['accuracy', 'f1_micro']
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

    p.getDC().save('sklearn_test_results')

if __name__ == '__main__':
    RunML()
