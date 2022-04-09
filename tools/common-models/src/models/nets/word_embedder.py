import numpy as np

from src.common.singleton import SingletonFactory
from src.configuration.settings_template import Settings
from src.models.nets.utils import Embedding


class WordEmbedder(metaclass=SingletonFactory):

    def __init__(self):
        self.embedding_length = Settings.WORD_EMBEDDINGS.MAX_NUM_WORDS_IN_SEQUENCE
        self.glove_file = Settings.WORD_EMBEDDINGS.GLOVE.USE_PRETRAINED_WORD_VECTORS_FROM_FILE
        self.trainable = Settings.WORD_EMBEDDINGS.GLOVE.WORD_VECTORS_ARE_TRAINABLE

        self.embedding_dim = Settings.WORD_EMBEDDINGS.EMBEDDING_VECTOR_SIZE
        self.max_num_words = Settings.WORD_EMBEDDINGS.MAX_NUM_WORDS_IN_VOCABULARY

    def __hash__(self):
        return hash(self.embedding_length) # TODO: Check change

    @staticmethod
    def create():
        if not Settings.WORD_EMBEDDINGS.GLOVE.USE_PRETRAINED_WORD_VECTORS_FROM_FILE:
            return NotPretrained()
        else:
            return Glove()

class Glove(WordEmbedder):

    # Get the file to use for embeddings, can either be a generic file downloaded from gloVe, or a custom trained file
    def get_glove_file(self):
        return open(self.glove_file, encoding="utf8")


    # Code is from https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
    def get_embedding_matrix(self, tokenizer):
        embeddings_index = {}
        with self.get_glove_file() as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))

        word_index = tokenizer.word_index

        # prepare embedding matrix
        num_words = min(self.max_num_words, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, self.embedding_dim))
        for word, i in word_index.items():
            if i >= self.max_num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return num_words, embedding_matrix

    # Get embedding layer from embedding matrix, the first layer of the NN
    def get_embedding_layer(self, tokenizer):
        num_words, embedding_matrix = self.get_embedding_matrix(tokenizer)

        return Embedding(num_words,
                         self.embedding_dim,
                         weights=[embedding_matrix],
                         input_length=self.embedding_length,
                         trainable=self.trainable)



# When pretrained word embeddings are not to be used, get an embedding layer with randomly initialized weights to start
class NotPretrained(WordEmbedder):

    # def __init__(self, *args):N
    #     # Can't have the case where there was no pretraining but training is also not allowed.
    #     # We will never learn word embeddings as they will just keep their randomly initialized values.
    #     # if Settings.WORD_EMBEDDINGS.GLOVE.TRAINABLE_TYPE == SettingsEnumOptions.GloveTrainableTypes.NOT_TRAINABLE:
    #     #     raise SettingsConfigurationError
    #     super(NotPretrained, self).__init__(*args) # TODO: test args

    def get_embedding_layer(self, tokenizer):
        word_index = tokenizer.word_index
        num_words = min(self.max_num_words, len(word_index) + 1)

        return Embedding(num_words,
                         self.embedding_dim,
                         input_length=self.embedding_length,
                         trainable=self.trainable)
