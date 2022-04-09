# -*- coding: utf-8 -*-
import os
import sys
import pdb

import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from src.configuration.settings_template import Settings
from src.common import utils
import traceback


def get_saved_model_filename(model_run_instance, fold_num):
    if Settings.IO.SAVED_MODEL_DIR is None:
        directory = os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, "saved_models", model_run_instance.model_name, model_run_instance.label)
    else:
        directory = os.path.join(Settings.IO.SAVED_MODEL_DIR, model_run_instance.model_name, model_run_instance.label)
    utils.ensure_directory(directory)
    return os.path.join(directory, '{}-{}-{}.pkl'.format(model_run_instance.model_name, model_run_instance.label, fold_num))

def save_data_frame(X, y, y_label_name):
    out_mat = np.hstack((X,y.reshape(-1,1)))
    directory = os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, "saved_data_frame")
    utils.ensure_directory(directory)
    out_path = os.path.join(directory, "data_with_label_{}.csv".format(y_label_name))
    out_df = pd.DataFrame(data=out_mat, columns=X.columns.tolist()+[y_label_name])
    out_df.to_csv(out_path, header=True, index=False)
    return

def save_model(fitted_model, model_run_instance, fold_num):
    try:
        joblib.dump(fitted_model, get_saved_model_filename(model_run_instance, fold_num))
    except Exception as e:
        print("Could not save model {} for fold {}: {}".format(model_run_instance, fold_num, e))

def load_model_for_fold(model_run_instance, fold_num):
    model = None
    try:
        model = {'fold': fold_num, 'model': joblib.load(get_saved_model_filename(model_run_instance, fold_num))}
    except Exception as e:
        print("Could not load model {} for fold {}: {}".format(model_run_instance, fold_num, e))
    return model

def load_models(model_run_instance):
    models = []
    for fold_num in range(Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS):
        try:
            loaded_model = {'fold': fold_num, 'model': joblib.load(get_saved_model_filename(model_run_instance, fold_num))}
            models.append(loaded_model)
        except:
            traceback.print_exc()
            print("Failed to load model for fold {}".format(fold_num+1))

    return models

def load_model(model_run_instance, fold_num):
    return joblib.load(get_saved_model_filename(model_run_instance, fold_num))

def get_saved_weights_filename(model_run_instance, fold_num):
    directory = os.path.join(Settings.IO.RESULTS_OUTPUT_FOLDER, "saved_feature_weights", model_run_instance.model_name, model_run_instance.label)
    utils.ensure_directory(directory)
    return os.path.join(directory, '{}-{}-{}.csv'.format(model_run_instance.model_name, model_run_instance.label, fold_num))

def save_feature_weights(fitted_model, model_run_instance, fold_num):
    model_inst = fitted_model.named_steps['MODEL']
    if issubclass(type(model_inst), RandomForestRegressor):
        feature_weights = model_inst.feature_importances_
    elif issubclass(type(model_inst), RandomForestClassifier):
        feature_weights = model_inst.feature_importances_
    elif issubclass(type(model_inst), LogisticRegression):
        feature_weights = model_inst.coef_.flatten()
    else:
        sys.exit("SAVE_FEATURE_WEIGHTS was enabled, but the model %s is not yet supported. Please fix me!"%(fitted_model._name))

    # Get feature names
    feature_names = fitted_model.named_steps['EXTRACT_FEATURES'].transformer_list[0][1].steps[0][1].cols[:] # Regular features
    if len(feature_names) < len(feature_weights):
        lang_feat_names = fitted_model.named_steps['EXTRACT_FEATURES'].transformer_list[1][1].steps[1][1].get_feature_names()[:] # Language features
        feature_names.extend(lang_feat_names)

    feat_weight_df = pd.DataFrame(data={'Features': feature_names, 'Weights': feature_weights}, index=range(len(feature_names)))

    # BB - Remove duplicates!  I'm not sure why, but the sklearn pipeline does not collapse features with the same
    #      name across folds.
    feat_weight_df['UniqueFeatWeights'] = feat_weight_df.groupby(['Features'], sort=False)['Weights'].transform(sum)
    feat_weight_df = feat_weight_df.drop_duplicates(subset=['Features'])
    feat_weight_df = feat_weight_df[['Features','UniqueFeatWeights']]

    output_weights_filepath = get_saved_weights_filename(model_run_instance, fold_num)
    feat_weight_df.to_csv(output_weights_filepath, index=False, header=False)
    return
