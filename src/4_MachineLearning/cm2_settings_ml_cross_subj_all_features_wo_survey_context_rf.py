import os 
import numpy as np
import pandas as pd
import json
import pickle
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from commonmodels2.stages.pipeline import Pipeline
from commonmodels2.models.model import SklearnModel
from commonmodels2.stages.load_data import CSVLoaderStage
from commonmodels2.stages.preprocessing import ImputerPreprocessingStage, FeatureScalerPreprocessingStage, EncoderPreprocessingStage
from commonmodels2.stages.cross_validation import GenerateCVFoldsStage, CrossValidationStage, SupervisedCVContext
from commonmodels2.stages.training_stage import SupervisedTrainingContext
from commonmodels2.stages.evaluation_stage import EvaluationStage, SupervisedEvaluationContext

def my_rf_model_func(params):
    model = RandomForestRegressor()
    model.set_params(**params)
    return model

def RunML():
    # Create a Sklearn model using the custom model create function above
    skm = SklearnModel()
    skm.set_model_create_func(my_rf_model_func)
    skm.set_model_params({'n_estimators': 200, 'max_depth': 50})

    p = Pipeline()

    # Stage 0: Load data from CSV file
    s0 = CSVLoaderStage()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    s0.setFilePath(os.path.join(dir_path, os.pardir, '2_FeatureCorrectionAndFilter', 'results', 'merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_non_survey_context_feats.csv'))
    feat_cols = pd.read_csv(os.path.join(dir_path, 'all_feats_wo_context.txt'), header=None).values.flatten()
    label_cols = ['stress.d']
    p.addStage(s0)

    # Stage 1: Generate cross-validation folds
    s1 = GenerateCVFoldsStage(strategy='load_premade', strategy_args={'file_path': os.path.join(dir_path, os.pardir, '3_PrecomputedFolds', 'results', 'stratifiedOn-stress.d_percentileBins-10_shuffle-True_seed-3748_folds.csv')})
    p.addStage(s1)

    # Stage 2: Cross-validation
    s2 = CrossValidationStage()

    cv_context_skm = SupervisedCVContext()
    cv_context_skm.add_preprocessing_stage(ImputerPreprocessingStage(feat_cols, 'mean'))
    cv_context_skm.add_preprocessing_stage(FeatureScalerPreprocessingStage(feat_cols, 'standardize'))

    training_context = SupervisedTrainingContext()
    training_context.model = skm
    training_context.feature_cols = feat_cols
    training_context.label_cols = label_cols
    cv_context_skm.training_context = training_context

    eval_context = SupervisedEvaluationContext()
    eval_context.label_cols = label_cols
    eval_context.eval_funcs = [pearsonr, spearmanr, 'smape']
    cv_context_skm.eval_context = eval_context

    s2.setCVContext(cv_context_skm)
    p.addStage(s2)

    p.run()

    p.getDC().save('cm2_results_ncx_feats_cross_subj_rf')


if __name__ == '__main__':
    RunML()
