import os 
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from torch import nn
from commonmodels2.stages.pipeline import Pipeline
from commonmodels2.models.model import PyTorchModel
from commonmodels2.stages.load_data import CSVLoaderStage
from commonmodels2.stages.preprocessing import ImputerPreprocessingStage, FeatureScalerPreprocessingStage, EncoderPreprocessingStage
from commonmodels2.stages.cross_validation import GridParamSearch, GenerateCVFoldsStage, NestedCrossValidationStage, CrossValidationStage, NestedSupervisedCVContext, SupervisedCVContext, PyTorchModelTuningContext
from commonmodels2.stages.training_stage import SupervisedTrainingContext
from commonmodels2.stages.evaluation_stage import EvaluationStage, SupervisedEvaluationContext

def my_mlp_model_func(params):
    model = nn.Sequential()
    for layer_idx in range(params['num_layers']):
        model.add_module('linear'+str(layer_idx), nn.LazyLinear(params['layer_width']))
        model.add_module('relu'+str(layer_idx), params['activation_func']())

    model.add_module('linear'+str(params['num_layers']), nn.LazyLinear(1))
    #model.add_module('softmax', nn.Softmax())
    return model

def RunML():
    # Create a PyTorch model using the custom model create function above
    ptm = PyTorchModel()
    ptm.set_model_create_func(my_mlp_model_func)
    ptm.set_fit_params({'lr': 1e-3, 'batch_size':32, 'train_split': None, 'max_epochs': 500})
    ptm.set_criterion_params({'criterion': 'mean_squared_error'})
    ptm.set_model_params({'num_layers': 3, 'layer_width': 30, 'activation_func': nn.GELU})
    ptm.set_optimizer_params({'optimizer': 'adam'})

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

    # Stage 2: Nested cross-validation
    #s2 = NestedCrossValidationStage()
    s2 = CrossValidationStage()

    #cv_context_ptm = NestedSupervisedCVContext()
    cv_context_ptm = SupervisedCVContext()
    cv_context_ptm.add_preprocessing_stage(ImputerPreprocessingStage(feat_cols, 'mean'))
    cv_context_ptm.add_preprocessing_stage(FeatureScalerPreprocessingStage(feat_cols, 'standardize'))

    training_context = SupervisedTrainingContext()
    training_context.model = ptm
    training_context.feature_cols = feat_cols
    training_context.label_cols = label_cols
    cv_context_ptm.training_context = training_context

    #tuning_context = PyTorchModelTuningContext()
    #tuning_context.model_param_search = GridParamSearch({'num_layers': [1,2,3], 'layer_width': [10,20,30], 'activation_func': [nn.GELU]})
    #tuning_context.optimizer_params_search = GridParamSearch({"optimizer": ["sgd"]})
    #tuning_context.param_eval_func = spearmanr
    #tuning_context.param_eval_goal = 'max'
    #cv_context_ptm.tuning_context = tuning_context

    eval_context = SupervisedEvaluationContext()
    eval_context.label_cols = label_cols
    eval_context.eval_funcs = [pearsonr, spearmanr, 'smape']
    cv_context_ptm.eval_context = eval_context

    #cv_context_ptm.cv_folds_stage = GenerateCVFoldsStage(strategy='stratified_grouped', strategy_args={'num_folds': 3, 'stratify_on': 'stress.d', 'percentile_bins': 10, 'seed': 3748, 'group_by': 'snapshot_id'})
    #s2.set_nested_cv_context(cv_context_ptm)
    s2.setCVContext(cv_context_ptm)
    p.addStage(s2)

    p.run()

    p.getDC().save('cm2_results_ncx_feats_cross_subj_mlp')


if __name__ == '__main__':
    RunML()
