# -*- coding: utf-8 -*-
import abc

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.utils import shuffle
from src.common import utils
from src.io.read_data_input import Dataset
from src.configuration.settings_template import Settings
from src.io.print_output import Print
from src.metrics.score import Score
from src.pipeline.cross_validation.cv_param_grid import CVParamGrid
from src.pipeline.pipeline import create_pipeline
from src.data.utils import get_k_folds, get_group_k_folds, get_stratified_k_folds, get_stratified_group_k_folds


class CrossValidation(metaclass=abc.ABCMeta):
    def __init__(self, X, y, model_run_instance, fold_num, param_grid):
        #self.X, self.y = shuffle(X, y) # BB - Why is this shuffling without checking settings?
        self.X = X
        self.y = y
        self.model_run_instance = model_run_instance
        self.fold_num = fold_num
        self.param_grid = param_grid


    @staticmethod
    def get_cross_validated_model(X, y, model_run_instance, fold_num):

        print("Running cross validation for ", model_run_instance)
        param_grid = CVParamGrid(model_run_instance) # TODO: How to get hierarchical model params

        model_type = NoCrossValidationModel if param_grid.has_only_one_combination_of_parameters else CrossValidatedModel
        return model_type(X, y, model_run_instance, fold_num, param_grid).get()




class NoCrossValidationModel(CrossValidation):

    def get(self):
        print(
            "Only one combination of parameters given. Not running GridSearchCV and instead just fitting with these parameters.")

        param_dict = self.param_grid.as_single_dict
        p = create_pipeline(self.X, self.model_run_instance, self.fold_num, param_dict)
        p.set_params(**param_dict)

        if self.param_grid.has_no_cv_params_for_model(self.model_run_instance):
            p.named_steps.MODEL.set_params(**({}))

        Print.print_hyperparameters_for_fold(self.model_run_instance, self.fold_num, param_dict,
                                             ran_cross_validation=False)
        Print.print_pipeline_params(p, param_dict)


        p.fit(self.X, self.y)

        if utils.class_has_method(p, 'do_after_fit'): # TODO: Check
            p.do_after_fit()

        return p

class CrossValidatedModel(CrossValidation):

    def _get_grid_search_params(self):
        scoring_function = Score.get_scorer_for_cv()

        print("Using {} to score cross validation".format(scoring_function))

        return  { 'scoring': scoring_function } # Can add any other params as necessary

    def _get_cv_folds(self):
        if Settings.CROSS_VALIDATION.STRATIFIED_SPLIT_CLASSES is not None and len(Settings.CROSS_VALIDATION.STRATIFIED_SPLIT_CLASSES) > 0:
            X, y = Dataset().apply_column_filters(self.X, self.y, Settings.CROSS_VALIDATION.STRATIFIED_SPLIT_CLASSES)

            k = Settings.CROSS_VALIDATION.NUM_CV_TRAIN_VAL_FOLDS
            if Settings.CROSS_VALIDATION.STRATIFIED_BINNING_EDGES is not None:
                split = Settings.CROSS_VALIDATION.STRATIFIED_BINNING_EDGES
            else:
                split = Settings.CROSS_VALIDATION.STRATIFIED_NUMBER_PERCENTILE_BINS
            do_shuffle = Settings.CROSS_VALIDATION.SHUFFLE
            rand_seed = Settings.RANDOM_STATE
            if Settings.COLUMNS.GROUP_BY_COLUMN:
                group_by = Settings.COLUMNS.GROUP_BY_COLUMN
                train_test_index_list = get_stratified_group_k_folds(X, y, k, group_by, split, do_shuffle, rand_seed)
            else:
                train_test_index_list = get_stratified_k_folds(X, y, k, split, do_shuffle, rand_seed)
        else:
            X = self.X
            if Settings.CROSS_VALIDATION.SHUFFLE:
                X = X.sample(frac=1).reset_index(drop=True)

            k = Settings.CROSS_VALIDATION.NUM_CV_TRAIN_VAL_FOLDS
            if Settings.COLUMNS.GROUP_BY_COLUMN:
                group_by = Settings.COLUMNS.GROUP_BY_COLUMN
                train_test_index_list = get_group_k_folds(X, k, group_by)
            else:
                train_test_index_list = get_k_folds(X, k)
            
        return train_test_index_list

    def get(self):

        print("Performing grid search...")

        param_list = self.param_grid.as_list
        grid_search_params = self._get_grid_search_params()
        p = create_pipeline(self.X, self.model_run_instance, self.fold_num)
        cv = self._get_cv_folds()
        grid_search = GridSearchCV(estimator=p, param_grid=param_list, refit=True, cv=cv, n_jobs=-1, verbose=1, **grid_search_params)

        print("param_grid:")
        print(param_list)

        # file_utils.ensure_directory("logs")
        # f = open('logs/output-{}-{}.txt'.format(model.__name__, label), "w")
        # original_stderr = sys.stderr
        # original_stdout = sys.stdout
        # sys.stderr = f
        # sys.stdout = f
        grid_search.fit(self.X, self.y)
        # sys.stderr = original_stderr
        # sys.stdout = original_stdout
        Print.print_hyperparameters_for_fold(self.model_run_instance, self.fold_num, grid_search, ran_cross_validation=True)

        return grid_search.best_estimator_
