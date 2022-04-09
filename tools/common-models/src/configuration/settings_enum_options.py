import os
from enum import auto

from src.common.meta import CommonEnumMeta


class SettingsEnumOptions:
    class Prediction(CommonEnumMeta):
        CLASSIFICATION = auto()
        REGRESSION = auto()
        MULTICLASS = auto()

        def is_regression(self):
            return self == type(self).REGRESSION

        def is_classification(self):
            return self == type(self).CLASSIFICATION or self == type(self).MULTICLASS

        def is_multiclass(self):
            return self == type(self).MULTICLASS


    class FeatureInput(object):
        includes_language_features = False
        includes_regular_features = False
        includes_word_embeddings = False


    class LanguageFeatureInput(FeatureInput):
        includes_language_features = True

        @classmethod
        def with_language_from_column(cls, column_name):
            cls.language_column_name = column_name
            return cls

    class RegularFeatureInput(FeatureInput):
        includes_regular_features = True

        @classmethod
        def with_regular_features(cls, feature_list):
            cls.regular_feature_names = feature_list
            return cls

        @classmethod
        def with_regular_features_from_file(cls, filename):
            with open(os.path.join(filename), "r") as f:
                feature_names = [line.strip() for line in f]

            cls.regular_feature_names = feature_names
            return cls

    class WordEmbeddingsInput(LanguageFeatureInput):
        includes_word_embeddings = True
        includes_language_features = False

    class CombineLanguageAndRegularFeatureInput(RegularFeatureInput, LanguageFeatureInput):
        pass

    class Tasks(CommonEnumMeta):
        MODEL_COMPARISON = auto()
        CREATE_TRAIN_TEST_FOLDS = auto()

    class ResamplingTypes(CommonEnumMeta):
        BASELINE_DATASET_SAMPLER = auto()
        BASIC_UNDERSAMPLING = auto()
        BASIC_OVERSAMPLING = auto()
        BASIC_OVERSAMPLING_3D = auto()
        BASIC_UNDERSAMPLING_3D = auto()
        SMOTE_SAMPLER = auto()
        BAGGING_CLF = auto()
