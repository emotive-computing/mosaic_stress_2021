# -*- coding: utf-8 -*-
from keras.layers import Bidirectional

from src.models.nets.mixins import RnnMixin
from src.models.nets.utils import Sequential, Dense, BatchNormalization, Dropout
from .base_net import BaseNet


class SingleContextRnn(RnnMixin, BaseNet):

    def build_model(self, X):
        self.model = Sequential()

        if self.feature_source.includes_word_embeddings:
            self.model.add(self.get_embedding_layer())

        else:
            raise NotImplementedError("SingleContextRnn currently only supports using word embeddings "
                                      "(finds context within each instance of text using its embedded sequence of words")

        rnn_layer = self.rnn_type(self.units, dropout=self.dropout_rate,
                                  recurrent_dropout=self.dropout_rate, return_sequences=False)

        if (self.rnn_is_bidirectional):
            rnn_layer = Bidirectional(rnn_layer)

        self.model.add(rnn_layer)

        if self.use_batch_normalization:
            self.model.add(BatchNormalization())

        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(self.n_outputs, activation=self.final_activation))  # try relu, leaky relu

        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.training_metrics)

        return self.model


