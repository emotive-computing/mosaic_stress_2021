from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Dataset sampler similar to those for rebalancing the dataset to have even classes
# However, in the event we want to compare no resampling against other resampling methods, it is convenient to have this class to use in the scikit-learn pipeline
# This is because the sampler step in the pipeline will not accept a value of none
# This code is from: https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/examples/applications/plot_over_sampling_benchmark_lfw.py
from src.configuration.settings_template import SettingsEnumOptions

import numpy as np

balance_type_key = 'balance_type'

class DatasetSamplerMixin(object):
    def get_params(self, deep=True):
        return {
            balance_type_key: self.balance_type
        }

    def set_params(self, **parameters):
        self.__class__ =  parameters[balance_type_key] # get_resampler(parameters)
        #key = 'balance_type' if 'balance_type' in parameters else 'resampler__balance_type'
        self.__init__() #parameters[balance_type_key])

        return self


class DatasetSampler(DatasetSamplerMixin, object):
    def __init__(self, balance_type=None):
        self.balance_type = balance_type

    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self.fit(X, y)

    def fit_sample(self, X, y):
        return self.sample(X, y)

# Get dataset balance type to use in pipeline
def get_resampler(hyperparameters=None):
    # Get data resampling type to use from hyperparameters passed in
    # If no hyperparameters specified, use undersampling by default
 #   key = PipelineStages.RESAMPLER.replace('resampler__', '')

    balance_type = SettingsEnumOptions.ResamplingTypes.BASIC_UNDERSAMPLING \
        if not (hyperparameters and hyperparameters[balance_type_key]) \
        else hyperparameters[balance_type_key]

    # Return the right instance of the dataset sampler class based on balance_type
    if balance_type == SettingsEnumOptions.ResamplingTypes.BASIC_UNDERSAMPLING:
        return RUSResampler
    elif balance_type == SettingsEnumOptions.ResamplingTypes.BASIC_OVERSAMPLING:
        return ROSResampler
    elif balance_type == SettingsEnumOptions.ResamplingTypes.BASIC_OVERSAMPLING_3D:
        return ROSResampler3D
    elif balance_type == SettingsEnumOptions.ResamplingTypes.BASIC_UNDERSAMPLING_3D:
        return RUSResampler3D
    elif balance_type == SettingsEnumOptions.ResamplingTypes.SMOTE_SAMPLER:
        return SMOTEResampler
    elif balance_type == SettingsEnumOptions.ResamplingTypes.BASELINE_DATASET_SAMPLER \
            or balance_type == SettingsEnumOptions.ResamplingTypes.BAGGING_CLF:
        return DatasetSampler
    else:  # if value given in hyperparameters for balance type is unknown
        raise KeyError("Balance type not supported {}".format(balance_type))



#TODO: Would have been better to use a class decorator here to specify the type of base class to be used.
# Alternatively a decorator on the __init__ function could have been used.
# Using a metaclass or a class factory are additional options.
# However, was running into issues here and since these are just 3 very small wrapper classes, going to leave for now.
# But should be combined into a more generic class in the future.Nr
class RUSResampler(DatasetSamplerMixin, RandomUnderSampler):
    def __init__(self, balance_type=None, **kwargs):
        self.balance_type = balance_type
        super(RUSResampler, self).__init__(**kwargs)

class ROSResampler(DatasetSamplerMixin, RandomOverSampler):
    def __init__(self, balance_type=None, **kwargs):
        self.balance_type = balance_type
        super(ROSResampler, self).__init__(**kwargs)

class SMOTEResampler(DatasetSamplerMixin, SMOTE):
    def __init__(self, balance_type=None, **kwargs):
        self.balance_type = balance_type
        super(SMOTEResampler, self).__init__(**kwargs)


class ROSResampler3D(DatasetSamplerMixin, RandomOverSampler):
    def __init__(self, balance_type=None, **kwargs):
        self.balance_type = balance_type
        super(ROSResampler3D, self).__init__(**kwargs)

    def _size_2D(self, X, num_time_steps, num_features):
        return np.reshape(X, newshape=(-1, num_time_steps * num_features))

    def _size_3D(self, X, num_instances, num_time_steps, num_features):
        return np.reshape(X, newshape=(X.shape[0], num_time_steps, num_features))

    # def sample(self, X, y):
    #     num_instances, num_time_steps, num_features = X.shape
    #     return X, y

    def fit(self, X, y):
        num_instances, num_time_steps, num_features = X.shape
        Xt = self._size_2D(X, num_time_steps, num_features)
        Xf, Yf =  super(ROSResampler3D, self).fit(Xt, y)
        return self._size_3D(Xf, num_instances, num_time_steps, num_features), Yf

    def fit_sample(self, X, y):
        num_instances, num_time_steps, num_features = X.shape
        Xt = self._size_2D(X, num_time_steps, num_features)
        Xf, Yf =  super(ROSResampler3D, self).fit_sample(Xt, y)
        return self._size_3D(Xf, num_instances, num_time_steps, num_features), Yf
        #return self.sample(X, y)

    def fit_resample(self, X, y):
        num_instances, num_time_steps, num_features = X.shape
        Xt = self._size_2D(X, num_time_steps, num_features)
        Xf, Yf =  super(ROSResampler3D, self).fit_resample(Xt, y)
        return self._size_3D(Xf, num_instances, num_time_steps, num_features), Yf


class RUSResampler3D(DatasetSamplerMixin, RandomUnderSampler):
    def __init__(self, balance_type=None, **kwargs):
        self.balance_type = balance_type
        super(RUSResampler3D, self).__init__(**kwargs)

    def _size_2D(self, X, num_time_steps, num_features):
        return np.reshape(X, newshape=(-1, num_time_steps * num_features))

    def _size_3D(self, X, num_instances, num_time_steps, num_features):
        return np.reshape(X, newshape=(X.shape[0], num_time_steps, num_features))

    # def sample(self, X, y):
    #     num_instances, num_time_steps, num_features = X.shape
    #     return X, y

    def fit(self, X, y):
        num_instances, num_time_steps, num_features = X.shape
        Xt = self._size_2D(X, num_time_steps, num_features)
        Xf, Yf =  super(RUSResampler3D, self).fit(Xt, y)
        return self._size_3D(Xf, num_instances, num_time_steps, num_features), Yf

    def fit_sample(self, X, y):
        num_instances, num_time_steps, num_features = X.shape
        Xt = self._size_2D(X, num_time_steps, num_features)
        Xf, Yf =  super(RUSResampler3D, self).fit_sample(Xt, y)
        return self._size_3D(Xf, num_instances, num_time_steps, num_features), Yf
        #return self.sample(X, y)

    def fit_resample(self, X, y):
        num_instances, num_time_steps, num_features = X.shape
        Xt = self._size_2D(X, num_time_steps, num_features)
        Xf, Yf =  super(RUSResampler3D, self).fit_resample(Xt, y)
        return self._size_3D(Xf, num_instances, num_time_steps, num_features), Yf