
# Only allow certain static attributes to be set at initialization, as this will be passed from stage to stage of the pipeline
# And we don't want a given stage inadvertently modifying it
from collections import OrderedDict

from imblearn.ensemble import RUSBoostClassifier
from imblearn.pipeline import Pipeline as ImbLearnPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.configuration.settings_template import Settings
from src.models.nets.utils import Tokenizer
from src.pipeline.estimator_wrapper import EstimatorWrapper
from src.pipeline.extract_features import FeatureStep
from src.pipeline.stages import PipelineStages
from src.pipeline.transformers.transformer_wrapper import PipelineTransformerWrapper


class PipelineParams(object):


    def __init__(self, X, model_run_instance, fold_num, hyperparameters=None):
        self.model_run_instance = model_run_instance
        self.fold_num = fold_num
        self.hyperparameters = hyperparameters

        # Need tokenizer if using any neural nets with language modeling.
        # Otherwise can be set to none, just weird way of setting it just since slots are used.
        self.tokenizer = None
        if model_run_instance.feature_source.includes_word_embeddings:
            self.tokenizer = Tokenizer(num_words=Settings.WORD_EMBEDDINGS.MAX_NUM_WORDS_IN_VOCABULARY)
            self.tokenizer.fit_on_texts(X[model_run_instance.feature_source.language_column_name].values)

            self.word_embeddings_settings = Settings.WORD_EMBEDDINGS


    # Get the pipeline used by scikit learn


def create_pipeline(X, model_run_instance, fold_num, hyperparameters=None):
    current_pipeline = []

    # variables used as parameters for functions defining the stages of the pipeline
    params = PipelineParams(X, model_run_instance, fold_num, hyperparameters)

    # Defines the order of pipeline and the logic for which different stages of the pipeline are required
    pipeline_stages = OrderedDict({
        PipelineStages.EXTRACT_FEATURES: (True, FeatureStep.get_step),  # Call this feature input sources instead?
        PipelineStages.IMPUTER: (Settings.CROSS_VALIDATION.HYPER_PARAMS.IMPUTER, lambda _: PipelineTransformerWrapper()),
        PipelineStages.FEATURE_SCALER: (Settings.CROSS_VALIDATION.HYPER_PARAMS.FEATURE_SCALER, lambda _: PipelineTransformerWrapper()),
        PipelineStages.RESAMPLER: (Settings.PREDICTION.is_classification(), lambda _: PipelineTransformerWrapper()),
        PipelineStages.FEATURE_SELECTION: (Settings.CROSS_VALIDATION.HYPER_PARAMS.FEATURE_SELECTION, lambda _: PipelineTransformerWrapper()),
        PipelineStages.MODEL: (True, EstimatorWrapper.initialize_from_pipeline_params) # Always will at least have model in pipeline  lambda _: CalibratedClassifierCV(base_estimator=RandomForestClassifier(n_estimators=100)))
    })
#LinearSVC(class_weight='balanced'
    # # Adds each of the applicable stages as defined by pipeline_stages to the pipeline that will be used
    for key, (condition, callback) in pipeline_stages.items():
        if condition:
            current_pipeline.append((str(key), callback(params)))

    return Pipeline(current_pipeline) \
        if Settings.PREDICTION.is_regression() \
        else ImbLearnPipeline(current_pipeline)







