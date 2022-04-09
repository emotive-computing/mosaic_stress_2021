# -*- coding: utf-8 -*-

from keras import Input, Model
from keras.layers import TimeDistributed

from src.models.nets.mixins import SlidingWindowMixin, RnnMixin
from src.models.nets.utils import Sequential, Dense
from .base_net import BaseNet


class WindowRnn(RnnMixin, SlidingWindowMixin, BaseNet):

    def __init__(self, *args, **kwargs):
        kwargs['batch_size'] = 1
        super(WindowRnn, self).__init__(*args, **kwargs)

    def build_model(self, X):

        self.model = Sequential()

        # TODO: Fix for feature + language

        if self.feature_source.includes_word_embeddings:

            embedding_layer = self.get_embedding_layer()
            embedding_length = self.get_embedding_length()

            sentence_input = Input(shape=(embedding_length,), dtype='int32')
            embedded_sequences = embedding_layer(sentence_input)
            l_lstm = self.rnn_type(self.units, return_sequences=False)(embedded_sequences)
            sentEncoder = Model(sentence_input, l_lstm)

            review_input = Input(shape=(self.window_size, embedding_length), dtype='int32')
            review_encoder = TimeDistributed(sentEncoder)(review_input)
            l_sent = self.rnn_type(self.units, return_sequences=False)(review_encoder)
            preds = Dense(self.n_outputs, activation=self.final_activation)(l_sent)
            self.model = Model(review_input, preds)

        else:
            self.model.add(self.rnn_type(self.units, dropout=self.dropout_rate, return_sequences=False, batch_input_shape=(self.batch_size, self.window_size, X.shape[2]))) #, return_sequences=True)))
            self.model.add(Dense(self.n_outputs, activation=self.final_activation))  # try relu, leaky relu

        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.training_metrics)

        return self.model

