import numpy as np
from keras.layers import GRU
from keras.utils import to_categorical

from src.io.read_data_input import Dataset
from src.models.nets.utils import auc_roc, auprc
from src.models.nets.word_embedder import WordEmbedder


class SlidingWindowMixin(object):
    window_size = 3

class RnnMixin(object):
    rnn_type = GRU # default setting

    @classmethod
    def with_rnn_type(cls, rnn_type):  # Could be one of Kera's LSTM or GRU types, for example
        cls.rnn_type = rnn_type
        return cls

class NetMixin(object):
    pass

class BinaryClassificationNetMixin(NetMixin):
    @property
    def n_outputs(self):
        return 1

    @property
    def final_activation(self):
        return 'sigmoid'

    @property
    def loss(self):
        return 'binary_crossentropy'

    @property
    def training_metrics(self):
        return ['acc', auc_roc, auprc]

    def decision_function(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return np.array([[1 - i[0], i[0]] for i in self.model.predict(X)])

    def transform_y(self, y):
        return y

class MultiClassClassificationNetMixin(NetMixin):
    @property
    def n_outputs(self):
        le = Dataset().get_saved_label_encoder(self.label)
        return le.n_classes

    @property
    def final_activation(self):
        return 'softmax'

    @property
    def loss(self):
        return 'categorical_crossentropy'

    @property
    def training_metrics(self):
        return []

    def decision_function(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        # TODO: This isn't quite right. Fix.
        return np.array([[1 - i[0], i[0]] for i in self.model.predict(X)])

    def transform_y(self, y):
        le = Dataset().get_saved_label_encoder(self.label)
        return to_categorical(y, le.n_classes)


class RegressionNetMixin(NetMixin):
    @property
    def n_outputs(self):
        return 1

    @property
    def final_activation(self):
        return 'sigmoid'

    @property
    def loss(self):
        return 'mean_squared_error'

    @property
    def training_metrics(self):
        return []

    def decision_function(self, X):
        return self.predict_proba(X)

    def predict_proba(self, X):
        return np.array(self.model.predict(X).reshape(-1))

    def transform_y(self, y):
        return y


class WordEmbeddingsNetMixin(object):

    def get_embedding_layer(self):
        return self.embedder.get_embedding_layer(self.tokenizer)

    def get_embedding_dim(self):
        return self.embedder.embedding_dim

    def get_embedding_length(self):
        return self.embedder.embedding_length