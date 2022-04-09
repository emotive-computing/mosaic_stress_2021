import os 
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tensorflow import keras
import tensorflow as tf
from commonmodels2.stages.pipeline import Pipeline
from commonmodels2.models.model import TensorFlowModel
from commonmodels2.stages.load_data import CSVLoaderStage
from commonmodels2.stages.preprocessing import ImputerPreprocessingStage, FeatureScalerPreprocessingStage, EncoderPreprocessingStage
from commonmodels2.stages.cross_validation import GenerateCVFoldsStage, NestedCrossValidationStage, CrossValidationStage, NestedSupervisedCVContext, SupervisedCVContext
from commonmodels2.stages.training_stage import SupervisedTrainingContext
from commonmodels2.stages.evaluation_stage import EvaluationStage, SupervisedEvaluationContext
from commonmodels2.log.logger import Logger

predict_indices = None
def my_timeseries_transformer(X, y):
    global predict_indices

    Logger.getInst().info("Starting timeseries transform...")
    sequence_len = 3
    if y is not None:
        X.reset_index(drop=True, inplace=True)
        y_df = pd.DataFrame(data=y, columns=['labels'])
        merged_df = pd.concat((X, y_df), axis=1)
        grouped_df = merged_df.groupby('snapshot_id')
        new_X = None
        for key, group in grouped_df:
            group = group.drop(['snapshot_id'], axis=1)
            zero_pad_df = pd.DataFrame(0, index=range(-sequence_len+1,0), columns=group.columns)
            group = pd.concat((zero_pad_df, group), axis=0)
            group['labels'] = group['labels'].shift(-sequence_len+1) # Use the label at the most recent time in each seq
            ts_grouped_df = keras.utils.timeseries_dataset_from_array(data=group.iloc[:,:-1], targets=group.iloc[:,-1], sequence_length=sequence_len, batch_size=1)
            if new_X is None:
                new_X = ts_grouped_df
            else:
                new_X = new_X.concatenate(ts_grouped_df)
    else:
        predict_indices = []
        grouped_df = X.groupby('snapshot_id')
        indexed_ts_group_dfs = []
        new_X = None
        for key, group in grouped_df:
            group = group.drop(['snapshot_id'], axis=1)
            predict_indices.extend(group.index.tolist())
            zero_pad_df = pd.DataFrame(0, index=range(-sequence_len+1,0), columns=group.columns)
            group = pd.concat((zero_pad_df, group), axis=0)
            ts_grouped_df = keras.utils.timeseries_dataset_from_array(data=group, targets=None, sequence_length=sequence_len, batch_size=1)
            if new_X is None:
                new_X = ts_grouped_df
            else:
                new_X = new_X.concatenate(ts_grouped_df)
            #ts_grouped_dfs = [x for x in ts_grouped_df]
            #for group_idx in range(len(group_indices)):
            #    indexed_ts_group_dfs.append((group_indices[group_idx], tf.data.Dataset.from_tensors(ts_grouped_dfs[group_idx])))
            
        #sorted_indexed_ts_group_dfs = sorted(indexed_ts_group_dfs, key=lambda x: x[0]) 
        #new_X = None
        #for ts_group_idx, ts_group_df in sorted_indexed_ts_group_dfs:
        #    if new_X is None:
        #        new_X = ts_group_df
        #    else:
        #        new_X = new_X.concatenate(ts_group_df)
    Logger.getInst().info("... timeseries transform finished!")
    return new_X, None # new_X is a TF Dataset, so it contains a target "y" already

def my_prediction_transformer(y_pred):
    global predict_indices

    Logger.getInst().info("Transforming the predictions...")
    new_y_pred = y_pred[np.argsort(predict_indices)]
    Logger.getInst().info("... prediction transform finished!")
    return new_y_pred

def my_lstm_model_func(params):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(params['layer_width'], return_sequences=False))
    for layer_idx in range(params['num_layers']):
        model.add(keras.layers.Dense(params['layer_width'], activation=params['activation_func']))
    model.add(keras.layers.Dense(1))
    return model

def RunML():
    # Create a TensorFlow model using the custom model create function above
    tfm = TensorFlowModel()
    tfm.set_model_create_func(my_lstm_model_func)
    tfm.set_fit_transformer(my_timeseries_transformer)
    tfm.set_prediction_transformer(my_prediction_transformer)
    tfm.set_model_params({'num_layers': 3, 'layer_width': 10, 'activation_func': 'gelu'})
    tfm.set_fit_params({'batch_size': 32, 'validation_split': 0, 'epochs': 100})
    tfm.set_compile_params({'optimizer': 'sgd', 'loss': 'huber'})

    p = Pipeline()

    # Stage 0: Load data from CSV file
    s0 = CSVLoaderStage()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    s0.setFilePath(os.path.join(dir_path, os.pardir, '2_FeatureCorrectionAndFilter', 'results', 'merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_non_survey_context_feats.csv'))
    feat_cols = pd.read_csv(os.path.join(dir_path, 'all_feats_wo_context.txt'), header=None).values.flatten()
    feat_cols = np.append(feat_cols, ['snapshot_id']) # BB - this gets removed by the timeseries transformer
    label_cols = ['stress.d_shuffledWithinSubject']
    p.addStage(s0)

    # Stage 1: Generate cross-validation folds
    s1 = GenerateCVFoldsStage(strategy='load_premade', strategy_args={'file_path': os.path.join(dir_path, os.pardir, '3_PrecomputedFolds', 'results', 'stratifiedOn-stress.d_shuffledWithinSubject_percentileBins-10_shuffle-True_seed-3748_folds.csv')})
    p.addStage(s1)

    # Stage 2: Cross-validation
    s2 = CrossValidationStage()

    cv_context_tfm = SupervisedCVContext()
    cv_context_tfm.add_preprocessing_stage(ImputerPreprocessingStage(feat_cols, 'mean'))
    cv_context_tfm.add_preprocessing_stage(FeatureScalerPreprocessingStage(feat_cols, 'standardize'))

    training_context = SupervisedTrainingContext()
    training_context.model = tfm
    training_context.feature_cols = feat_cols
    training_context.label_cols = label_cols
    cv_context_tfm.training_context = training_context

    eval_context = SupervisedEvaluationContext()
    eval_context.label_cols = label_cols
    eval_context.eval_funcs = [pearsonr, spearmanr, 'smape']
    cv_context_tfm.eval_context = eval_context

    s2.setCVContext(cv_context_tfm)
    p.addStage(s2)

    # Stage 2: Nested cross-validation
    #s2 = NestedCrossValidationStage()
    #s2.addPreprocessingStage(ImputerPreprocessingStage(feat_cols, 'mean'))
    #s2.addPreprocessingStage(FeatureScalerPreprocessingStage(feat_cols, 'standardize'))
    #s2.setCVFoldsStage(GenerateCVFoldsStage(strategy='stratified_grouped', strategy_args={'num_folds': 2, 'stratify_on': 'stress.d', 'percentile_bins': 10, 'seed': 3748, 'group_by': 'snapshot_id'}))
    #cv_context_tfm = TensorFlowNestedSupervisedCVContext()
    #cv_context_tfm.model = tfm
    #cv_context_tfm.feature_cols = feat_cols
    #cv_context_tfm.y_label = label_cols
    ##cv_context_tfm.set_model_params({'num_layers': [1,2,3], 'layer_width': [10,20,30], 'activation_func': [nn.GELU]})
    #cv_context_tfm.set_model_params({'num_layers': [1], 'layer_width': [32], 'activation_func': ['relu']})
    #cv_context_tfm.param_eval_func = spearmanr
    #cv_context_tfm.param_eval_goal = 'max'
    #s2.setCVContext(cv_context_tfm)
    #eval_context = SupervisedEvaluationContext()
    #eval_context.y_label = label_cols
    #eval_context.eval_funcs = [spearmanr]
    #s2.setEvaluationContext(eval_context)

    p.run()

    p.getDC().save('cm2_results_ncx_feats_cross_subj_lstm3_shufwtn_baseline')


if __name__ == '__main__':
    RunML()
