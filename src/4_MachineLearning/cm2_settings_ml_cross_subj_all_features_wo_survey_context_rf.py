import os 
import numpy as np
import pandas as pd
import json
import pickle
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from commonmodels2.models.model import SklearnModel
from commonmodels2.stages.evaluation_stage import EvaluationStage, SupervisedEvaluationContext
from commonmodels2.stages.cross_validation import GenerateCVFoldsStage, NestedCrossValidationStage, CrossValidationStage, SklearnNestedSupervisedCVContext
from commonmodels2.stages.load_data import CSVDataLoaderStage
from commonmodels2.stages.pipeline import Pipeline
from commonmodels2.stages.preprocessing import ImputerPreprocessingStage, FeatureScalerPreprocessingStage, EncoderPreprocessingStage

def my_rf_model_func(params):
    model = RandomForestRegressor()
    model.set_params(**params)
    return model

def RunML():
    # Create a Sklearn model using the custom model create function above
    skm = SklearnModel()
    skm.set_model_create_func(my_rf_model_func)

    # Stage 0: Load data from CSV file
    s0 = CSVDataLoaderStage()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    s0.setFilePath(os.path.join(dir_path, os.pardir, '2_FeatureCorrectionAndFilter', 'results', 'merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_non_survey_context_feats.csv'))
    feat_cols = pd.read_csv(os.path.join(dir_path, 'all_feats_wo_context.txt'), header=None).values.flatten()
    label_cols = ['stress.d']

    # Stage 1: Generate cross-validation folds
    s1 = GenerateCVFoldsStage(strategy='load_premade', strategy_args={'file_path': os.path.join(dir_path, os.pardir, '3_PrecomputedFolds', 'results', 'stratifiedOn-stress.d_percentileBins-10_shuffle-True_seed-3748_folds.csv')})

    # Stage 2: Nested cross-validation
    s2 = NestedCrossValidationStage()
    s2.addPreprocessingStage(ImputerPreprocessingStage(feat_cols, 'mean'))
    s2.addPreprocessingStage(FeatureScalerPreprocessingStage(feat_cols, 'standardize'))
    s2.setCVFoldsStage(GenerateCVFoldsStage(strategy='stratified_grouped', strategy_args={'num_folds': 2, 'stratify_on': 'stress.d', 'percentile_bins': 10, 'seed': 3748, 'group_by': 'snapshot_id'}))
    cv_context_skm = SklearnNestedSupervisedCVContext()
    cv_context_skm.model = skm
    cv_context_skm.feature_cols = feat_cols
    cv_context_skm.y_label = label_cols
    #cv_context_skm.set_model_params({'n_estimators': [800], 'max_depth': [50]})
    cv_context_skm.set_model_params({'n_estimators': [200], 'max_depth': [50]})
    cv_context_skm.param_eval_func = spearmanr
    cv_context_skm.param_eval_goal = 'max'
    s2.setCVContext(cv_context_skm)
    eval_context = SupervisedEvaluationContext()
    eval_context.y_label = label_cols
    eval_context.eval_funcs = [spearmanr]
    s2.setEvaluationContext(eval_context)

    p = Pipeline()
    p.addStage(s0)
    p.addStage(s1)
    p.addStage(s2)
    p.run()

    nested_cv_results = p.getDC().get_item('nested_cv_results')

    for fold in nested_cv_results.keys():
        print(nested_cv_results[fold])

    if not os.path.isdir("cm2_results_ncx_feats_cross_subj_rf"):
        os.makedirs("cm2_results_ncx_feats_cross_subj_rf")
    with open(os.path.join("cm2_results_ncx_feats_cross_subj_rf", "results.json"), 'wb') as outfile:
    #    outfile.write(p.getDC().to_json())
        pickle.dump(p.getDC(), outfile)

if __name__ == '__main__':
    RunML()
