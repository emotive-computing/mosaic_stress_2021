from sklearn.pipeline import Pipeline, FeatureUnion

from src.models.nets.mixins import SlidingWindowMixin
from src.pipeline.concat_feature_union import ConcatFeatureUnion
from src.pipeline.transformers.selectors import PredictedItemSelector, WindowSelector, ItemSelector, TextSelector
from src.pipeline.stages import FeatureExtractionStages
from src.pipeline.transformers.pmi_count_vectorizer import PmiCountVectorizer
from src.pipeline.transformers.sequence_transformer import SequenceTransformer
from src.common.mixins import UseIncrementalPredictionsMixin
from sklearn.preprocessing import StandardScaler
from src.configuration.settings_template import Settings

class FeatureStep(object):

    @classmethod
    def get_step(cls, params):
        combined_features = []
        feature_source = params.model_run_instance.feature_source

        if feature_source.includes_regular_features:   
            itemList=feature_source.regular_feature_names
            if Settings.COLUMNS.GROUP_NORM_COLUMN:
                itemList=[Settings.COLUMNS.GROUP_NORM_COLUMN]+itemList
            selector = ItemSelector(itemList) \
                if not params.model_run_instance.model_class_is_subclass_of(SlidingWindowMixin) \
                else WindowSelector(feature_source.regular_feature_names, params.model_run_instance.model_class.window_size, True)

            combined_features.append(
                (
                    FeatureExtractionStages.FEATURES_FROM_DATA.name,
                    Pipeline([(FeatureExtractionStages.SELECT.name, selector)])
                )
            )

        if feature_source.includes_language_features:
            combined_features.append(
                (
                    FeatureExtractionStages.LANGUAGE.name,
                    Pipeline([
                        (FeatureExtractionStages.SELECT.name, cls.get_language_column_selector(feature_source, params)),
                        (FeatureExtractionStages.VECTORIZER.name,
                         PmiCountVectorizer.get(params.hyperparameters, params.model_run_instance.model_class_is_subclass_of(SlidingWindowMixin)))
                    ])
                )
            )

        if feature_source.includes_word_embeddings:
            window_size = None
            if params.model_run_instance.model_class_is_subclass_of(SlidingWindowMixin):
                window_size = params.model_run_instance.model_class.window_size

            combined_features.append(
                (
                    FeatureExtractionStages.WORD_EMBEDDINGS.name,
                    Pipeline([
                        (FeatureExtractionStages.SELECT.name, cls.get_language_column_selector(feature_source, params)),
                        (FeatureExtractionStages.SEQUENCE.name,
                         SequenceTransformer(tokenizer=params.tokenizer,
                                             word_embeddings_settings=params.word_embeddings_settings,
                                             use_sliding_window_size=window_size))
                    ])
                )
            )

        if params.model_run_instance.model_class_is_subclass_of(UseIncrementalPredictionsMixin) and params.model_run_instance.model_class.get_prop() > 0:  # feature_source.includes_predicted_values_of_other_predicted_vars():
            combined_features.append(
                (
                    FeatureExtractionStages.INCREMENTAL_PREDICTIONS.name,
                    Pipeline([
                        (
                        FeatureExtractionStages.SELECT.name, PredictedItemSelector(params.model_run_instance.model_class, params.model_run_instance.feature_source))
                    ])
                )
            )

        return cls.get_feature_union(params.model_run_instance, combined_features)

    @classmethod
    def get_feature_union(cls, model_run_instance, features):
        if model_run_instance.model_class_is_subclass_of(SlidingWindowMixin):
            return ConcatFeatureUnion(features)
        else:
            return FeatureUnion(features)

    @classmethod
    def get_language_column_selector(cls, feature_source, params):
        return TextSelector(feature_source.language_column_name) \
                if not params.model_run_instance.model_class_is_subclass_of(SlidingWindowMixin) \
                else WindowSelector([feature_source.language_column_name], params.model_run_instance.model_class.window_size)
