from sklearn.base import BaseEstimator, TransformerMixin

from src.configuration.settings_template import Settings
from src.models.nets.utils import sequence


# Transforms text strings into fixed-length arrays of numbers
# Words are mapped to numbers using tokenizer
class SequenceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, tokenizer, word_embeddings_settings, use_sliding_window_size=None):
        self.tokenizer = tokenizer
        self.word_embeddings_settings = word_embeddings_settings
        self.use_sliding_window_size = use_sliding_window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.use_sliding_window_size:
           # for x in X:
            X1 = [[i[0] for i in j] for j in X]

            a = [sequence.pad_sequences(self.tokenizer.texts_to_sequences(x), maxlen=self.word_embeddings_settings.MAX_NUM_WORDS_IN_SEQUENCE, padding='post', truncating='post') for x in X1]
            b = sequence.pad_sequences(a, maxlen=self.use_sliding_window_size, padding='pre')
            return b
        else:
            sequences = self.tokenizer.texts_to_sequences(X)
            return sequence.pad_sequences(sequences, maxlen=self.word_embeddings_settings.MAX_NUM_WORDS_IN_SEQUENCE, padding='post', truncating='post')
