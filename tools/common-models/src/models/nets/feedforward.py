# -*- coding: utf-8 -*-

from src.models.nets.utils import Sequential, Dense, Activation, Flatten, BatchNormalization, Dropout, maxnorm
from .base_net import BaseNet

class Feedforward(BaseNet):

    def build_model(self, X):
        self.model = Sequential()

        if self.feature_source.includes_word_embeddings:
            self.model.add(self.get_embedding_layer())
            self.model.add(Flatten())
            input_shape = (self.get_embedding_dim(),)
        else:
            input_shape = (X.shape[1],)

        self.model.add(
            Dense(self.units, input_shape=input_shape,
                  kernel_initializer=self.init_mode,
                  kernel_constraint=maxnorm(self.weight_constraint)),
        )
        self.model.add(Activation(self.activation1))
        # for i in range(1, self.num_hidden_layers):   # removed for now
        #     self.model.add(Dense(self.units))
        if self.use_batch_normalization:
            self.model.add(BatchNormalization())

        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(self.n_outputs, activation=self.final_activation))  # try relu, leaky relu

        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.training_metrics)

        return self.model
