# -*- coding: utf-8 -*-
import abc

from src.configuration.settings_template import Settings
from src.models.nets.mixins import BinaryClassificationNetMixin, RegressionNetMixin, \
    WordEmbeddingsNetMixin, NetMixin


# Determine if the model running is a neural network
from src.models.nets.word_embedder import WordEmbedder


def is_net(model):
    return issubclass(model, BaseNet)

class BaseNet(object):

    @staticmethod
    def get_prediction_type_base(prediction):
        if prediction.is_classification():
           # if le.is_binary_prediction:
            mixin = BinaryClassificationNetMixin

           # TODO: Fix for multi class
            # else:
            #     mixin = MultiClassClassificationNetMixin
        else:
            mixin = RegressionNetMixin
        return mixin

    @classmethod
    def get_base_class_type(cls, prediction, model_run_instance_wrapper, tokenizer=None):
        mixin = BaseNet.get_prediction_type_base(prediction)

        if model_run_instance_wrapper.type_.feature_source.includes_word_embeddings: # TODO: Fix
            bases = (cls, WordEmbeddingsNetMixin, mixin)
            dynamic_attrs = {'embedder': WordEmbedder.create(), 'tokenizer': tokenizer}
        else:
            bases = (cls, mixin)
            dynamic_attrs = {}

        return type(cls.__name__ + "-" + mixin.__name__, bases, dynamic_attrs)


    def __init__(self, label, feature_source,
                 epochs=12, batch_size=32,
                 units=75, optimizer="Adam",
                 init_mode='glorot_uniform', activation1='relu', dropout_rate=0.2,
                 weight_constraint=2,
                 use_batch_normalization=False, rnn_is_bidirectional=False):

        self.label = label
        self.is_multi_output = True if isinstance(label, list) else False
        self.feature_source = feature_source
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.optimizer = optimizer
        self.init_mode = init_mode
        self.activation1 = activation1
        self.dropout_rate = dropout_rate
        self.weight_constraint = weight_constraint
        self.use_batch_normalization = use_batch_normalization
        self.rnn_is_bidirectional = rnn_is_bidirectional

    @abc.abstractmethod
    def build_model(self, *args):
        pass

    def get_params(self, deep=True):
        return {
            'label': self.label,
            'feature_source': self.feature_souce,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'units': self.units,
            'optimizer': self.optimizer,
            'init_mode': self.init_mode,
            'activation1': self.activation1,
            'weight_constraint': self.weight_constraint,
            'dropout_rate': self.dropout_rate,
            'use_batch_normalization': self.use_batch_normalization,
            'rnn_is_bidirectional': self.rnn_is_bidirectional
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X):
        return self.decision_function(X)

    def fit(self, X, y):
        y = self.transform_y(y)
        self.build_model(X).fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
