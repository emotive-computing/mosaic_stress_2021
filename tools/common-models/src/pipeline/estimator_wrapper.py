

# Get the instance of the model to be used as a classifier in the pipeline
from collections import defaultdict

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.base import ClassifierMixin, BaseEstimator

# Wrapper around scikit estimators for pipeline so that if we need to use a Keras classifier or some other different classifier
# these can all be made interchangeable
from sklearn.calibration import CalibratedClassifierCV

from src.common.descriptors import SelfTypeWrapper
from src.configuration.settings_template import Settings, SettingsEnumOptions
from src.models.hierarchical_classifier import HierarchicalClassifier
from src.models.nets import base_net
from src.pipeline import resampling
from src.pipeline.stages import PipelineStages


class EstimatorWrapper(BaseEstimator, ClassifierMixin):

    def _build_args_from_hyperparameters(self):
        args = defaultdict()
        if self.hyperparameters:
            clf_prefix = PipelineStages.MODEL.get_prefix()
            for param in [param for param in self.hyperparameters.keys() if
                          param.startswith(clf_prefix) and not param.endswith(resampling.balance_type_key)]:
                args[param.replace(clf_prefix, "")] = self.hyperparameters[param]
        return args


    def __init__(self, model_run_instance_wrapper, tokenizer, fold_num, hyperparameters):
        self.model_run_instance_wrapper= model_run_instance_wrapper
        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer
        self.fold_num = fold_num

        kwargs = self._build_args_from_hyperparameters()

        model_class = self.model_run_instance_wrapper.type_.model_class


        if base_net.is_net(model_class):
            base_type = model_class.get_base_class_type(Settings.PREDICTION, model_run_instance_wrapper, tokenizer)
            self.classifier = base_type(self.model_run_instance_wrapper.type_.label, self.model_run_instance_wrapper.type_.feature_source, **kwargs)

        elif issubclass(model_class, HierarchicalClassifier):
            self.classifier = model_class(self.model_run_instance_wrapper.type_.label, fold_num)
        else:
            self.classifier = model_class(**kwargs)

    @classmethod
    def initialize_from_pipeline_params(cls, params):
        return cls(SelfTypeWrapper(params.model_run_instance), params.tokenizer, params.fold_num, params.hyperparameters)

    def fit(self, X, y, **fit_params):
        pass

    # def decision_function(self, X):
    #     pass


    def predict_proba(self, X):
        pass

    def predict(self, X):
        pass

    def predict_log_proba(self, X):
        pass

    def get_params(self, deep=True):
        params = {
            'hyperparameters': self.hyperparameters,
            'tokenizer': self.tokenizer,
            'model_run_instance_wrapper': self.model_run_instance_wrapper,
            'fold_num': self.fold_num
        }

        # This line is important! (Versus just always setting params['classes_]=self.classes_
        # This is because at least one of the models being used currently has a property called classes_ regardless if regression is run,
        # And if we set it here, we'll override it and cause problems
        # if hasattr(self, 'classes_'):
        #     params['classes_'] = self.classes_

        return params

    def set_params(self, **parameters):



        clf = self.classifier
        clf_params = {k: parameters[k] for k in parameters if not k.endswith(resampling.balance_type_key)}
        clf.set_params(**clf_params)
        if resampling.balance_type_key in parameters and parameters[resampling.balance_type_key] == SettingsEnumOptions.ResamplingTypes.BAGGING_CLF:
            self.__class__ = BalancedBaggingClassifier
            self.__init__(base_estimator=clf)
        else:
            self.__class__ = self.classifier.__class__
            self.__dict__.update(clf.__dict__)

        return self


