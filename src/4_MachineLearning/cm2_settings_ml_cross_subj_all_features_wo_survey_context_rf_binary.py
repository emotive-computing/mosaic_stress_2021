import os 
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from commonmodels2.stages.pipeline import Pipeline
from commonmodels2.models.model import SklearnModel
from commonmodels2.stages.load_data import CSVLoaderStage
from commonmodels2.stages.preprocessing import ImputerPreprocessingStage, FeatureScalerPreprocessingStage, EncoderPreprocessingStage
from commonmodels2.stages.cross_validation import GridParamSearch, GenerateCVFoldsStage, NestedCrossValidationStage, CrossValidationStage, NestedSupervisedCVContext, SklearnModelTuningContext
from commonmodels2.stages.training_stage import SupervisedTrainingContext
from commonmodels2.stages.evaluation_stage import SupervisedEvaluationContext

def my_sklearn_model_func(params):
    model = RandomForestClassifier()
    #model = KNeighborsClassifier()
    #model = SVC()
    model.set_params(**params)
    return model

def RunML():
    # Create a Sklearn model using the custom model create function above
    skm = SklearnModel()
    skm.set_model_create_func(my_sklearn_model_func)

    # Stage 0: Load data from CSV file
    s0 = CSVLoaderStage()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    s0.setFilePath(os.path.join(dir_path, os.pardir, '2_FeatureCorrectionAndFilter', 'results', 'merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_non_survey_context_feats.csv'))
    feat_cols = pd.read_csv(os.path.join(dir_path, 'all_feats_wo_context.txt'), header=None).values.flatten()
    label_cols = ['stress.d']

    # Stage 1: Binarize the stress labels per participant
    s1 = EncoderPreprocessingStage(['snapshot_id']+label_cols, 'binarize_median', {'group_by': 'snapshot_id'})

    # Stage 2: Generate cross-validation folds
    s2 = GenerateCVFoldsStage(strategy='load_premade', strategy_args={'file_path': os.path.join(dir_path, os.pardir, '3_PrecomputedFolds', 'results', 'stratifiedOn-stress.d_percentileBins-10_shuffle-True_seed-3748_folds.csv')})


    # Stage 3: Nested cross-validation
    s3 = NestedCrossValidationStage()
    ncv_context = NestedSupervisedCVContext()
    ncv_context.add_preprocessing_stage(ImputerPreprocessingStage(feat_cols, 'mean'))
    ncv_context.add_preprocessing_stage(FeatureScalerPreprocessingStage(feat_cols, 'standardize'))

    training_context = SupervisedTrainingContext()
    training_context.model = skm
    training_context.feature_cols = feat_cols
    training_context.label_cols = label_cols
    ncv_context.training_context = training_context

    tuning_context = SklearnModelTuningContext()
    tuning_context.model_param_search = GridParamSearch({'n_estimators': [30, 60, 100, 120], 'max_depth': [30, 60, 100, 120, 150]})
    #tuning_context.model_param_search = GridParamSearch({'n_neighbors':[3,4,5,8,12]})
    #tuning_context.model_param_search = GridParamSearch({'gamma':[2], 'C': [1e-4, 1e-3, 1e-2, 1e-1, 1]})
    tuning_context.param_eval_func = 'f1'
    tuning_context.param_eval_goal = 'max'
    ncv_context.tuning_context = tuning_context

    eval_context = SupervisedEvaluationContext()
    eval_context.label_cols = label_cols
    eval_context.eval_funcs = ['f1', 'accuracy', 'precision', 'recall']
    ncv_context.eval_context = eval_context

    cv_folds_stage = GenerateCVFoldsStage(strategy='stratified_grouped',
                                          strategy_args={'num_folds': 3,
                                                         'stratify_on': 'stress.d',
                                                         'bin_edges': [0.5],
                                                         'seed': 3748,
                                                         'group_by': 'snapshot_id'})
    ncv_context.cv_folds_stage = cv_folds_stage
    s3.set_nested_cv_context(ncv_context)

    p = Pipeline()
    p.addStage(s0)
    p.addStage(s1)
    p.addStage(s2)
    p.addStage(s3)
    p.run()

    p.getDC().save('cm2_results_ncx_feats_cross_subj_rf_binary')


if __name__ == '__main__':
    RunML()
