import itertools
import operator
import os
from functools import reduce
import numbers
from src.pipeline.stages import PipelineStages
from src.models.nets.utils import sequence
import nltk
from nltk.collocations import *
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.metrics.association import QuadgramAssocMeasures
from nltk.probability import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.configuration.settings_template import Settings


from src.common import utils

stemmer = SnowballStemmer("english", ignore_stopwords=True)

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

# simple way to list the nltk classes required to compute locations for different ngrams
# lists both AssocMeasures class and CollocationFinder
ngram_collocation_class_mappings = {
    '2': (BigramAssocMeasures, BigramCollocationFinder),
    '3': (TrigramAssocMeasures, TrigramCollocationFinder),
    '4': (QuadgramAssocMeasures, QuadgramCollocationFinder)
}

def find_ngrams_above_pmi(n, documents, min_pmi):

    collocation_classes = ngram_collocation_class_mappings[str(n)]
    ngram_measures = collocation_classes[0]()
    finder = collocation_classes[1].from_documents(documents)

    return sorted(finder.above_score(ngram_measures.pmi, min_pmi))

def get_ngram_pmi_vocab(n, documents, min_pmi):
    vocab = find_ngrams_above_pmi(n, documents, min_pmi)
    vocab_iter = [" ".join(ngram) for ngram in vocab]
    return vocab_iter

def get_unigram_vocab(documents):
    unigrams = []
    words = [item for items in documents for item in items] # flatten documents into list of words
    freq = FreqDist(words)

    for k in freq.keys():
        unigrams.append(k)
    return unigrams

def stem_and_remove_stop_words(documents, stop_words, use_stemming, strip_non_alphabetical):
    if strip_non_alphabetical:
        documents = [[w for w in d_tokens if w.isalpha()] for d_tokens in documents]

    if stop_words is not None:
        documents = [[w for w in d_tokens if w not in stop_words] for d_tokens in documents]

    if use_stemming:
        documents = [[stemmer.stem(w) for w in d_tokens] for d_tokens in documents]

    return documents

def get_all_ngrams_vocab(ngram_range, documents, min_pmi, exclude_language_features_set):
    start = int(ngram_range[0])
    vocab_iter = []

    # Unigrams are treated differently as we don't use pmi or word collocations, only check cutoff value for frequencies
    if start == 1:
        vocab_iter.append(get_unigram_vocab(documents))
        start += 1

    for i in range(start, int(ngram_range[1])+1):
        ngram_vocab = get_ngram_pmi_vocab(i, documents, min_pmi)
        vocab_iter.append(ngram_vocab)

    all_ngrams_gendered_excluded=[]
    all_ngrams=reduce(operator.add, vocab_iter) # concat all lists together into 1 list
    for ngram in all_ngrams:
        if ngram not in exclude_language_features_set:
            all_ngrams_gendered_excluded.append(ngram)
    return all_ngrams_gendered_excluded


# def get_text_from_df_records(df_records):
#     return [doc[Settings.COLUMNS.LANGUAGE_FEATURE] for doc in df_records]

# Extends the functionality of scikit-learn's CountVectorizer by also filtering n-grams by minimum PMI value passed in as a parameter
class PmiCountVectorizer(TfidfVectorizer):

    def __init__(self, input='content', encoding='utf-8', min_pmi=2.0,
                 decode_error='ignore', strip_accents=None, strip_non_alphabetical=True,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 2), analyzer='word', min_df=.01, use_stemming=True, use_idf=True, use_3d=False):
        TfidfVectorizer.__init__(self, input=input, encoding=encoding,
                                 decode_error=decode_error, strip_accents=strip_accents,
                                 lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer,
                                 stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range,
                                 analyzer=analyzer, min_df=min_df, use_idf=use_idf) # doesn't really matter if min_df is passed, it won't be used by CountVect as vocab is already fixed
        self.min_pmi = min_pmi
        self.use_stemming = use_stemming
        self.use_3d = use_3d
        self.strip_non_alphabetical = strip_non_alphabetical
		
        with open(Settings.IO.EXCLUDE_LANGUAGE_FEATURES_FILE) as f:
            exclude_language_features = [line.rstrip() for line in f]
        self.exclude_language_features_set=set(exclude_language_features)

        if self.ngram_range[0] < 1 or self.ngram_range[1] > 4:
            raise ValueError("n-gram range values not supported, must be in the range of 1 - 4")

    # Get vectorizer to use in pipeline
    # Transforms text into a matrix of n-grams (features) and their frequencies
    # Also filters out certain n-grams from the vocabulary based on criteria specified when finding this matrix
    @classmethod
    def get(cls, hyperparameters=None, is_3d=False):
        args = dict()

        if hyperparameters:  # Get parameters based on hyperparameters if passed in
            keys = [key for key in Settings.CROSS_VALIDATION.HYPER_PARAMS.VECTORIZER.keys() if key in hyperparameters]
            for key in keys:
                args[utils.remove_prefix(key, PipelineStages.EXTRACT_FEATURES.get_prefix() + "lang__language")] = \
                hyperparameters[key]

        if is_3d:
            args['use_3d'] = True

        # Construct a new instance of the PmiCountVectorizer based on the hyperparameters specified
        return cls(**args)

    def fit(self, raw_documents, y=None):
        if self.use_3d:
            # num_instances, num_time_steps, num_features = raw_documents.shape
            # Xt = np.reshape(raw_documents, newshape=(-1, num_time_steps * num_features))
            Xt = [t[len(t)-1][0] for t in raw_documents]
            return super(PmiCountVectorizer, self).fit(Xt)
        else:
            tokenized_docs = self.tokenize_all_documents(raw_documents)
            filtered_ngrams = get_all_ngrams_vocab(self.ngram_range, tokenized_docs, self.min_pmi, self.exclude_language_features_set)
            self.vocabulary = filtered_ngrams

            # filter stopwords, min max df, etc.
            super(TfidfVectorizer, self)._validate_vocabulary()
            max_df = self.max_df
            min_df = self.min_df
            max_features = self.max_features
            vocabulary, X = super(TfidfVectorizer, self)._count_vocab(raw_documents, True)
            if self.binary:
                X.data.fill(1)
            X = super(TfidfVectorizer, self)._sort_features(X, vocabulary)
            n_doc = X.shape[0]
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * n_doc)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            X, self.stop_words_ = super(TfidfVectorizer, self)._limit_features(X, vocabulary,
                                                       max_doc_count,
                                                       min_doc_count,
                                                       max_features)
            self.vocabulary_ = vocabulary
            self.vocabulary = vocabulary
            self._tfidf.fit(X)
            return self

    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents, y).transform(raw_documents)

    def transform(self, raw_documents, copy=True):
        if self.use_3d:
            Xt = [[i[0] for i in j] for j in raw_documents]
            Xt = [super(PmiCountVectorizer, self).transform(t).toarray() for t in Xt]
            return sequence.pad_sequences(Xt, maxlen=Xt[0].shape[0], padding='pre')
        else:
            return super(PmiCountVectorizer, self).transform(raw_documents)

    def tokenize_documents(self, raw_document):
        stop_words = self.get_stop_words()
        preprocess = self.build_preprocessor()
        tokenized_docs = [sent_detector.tokenize(preprocess(self.decode(doc)).strip(), realign_boundaries=False) for doc in [raw_document]]
        tokenized_docs = [word_tokenize(sent) for sent in itertools.chain.from_iterable(tokenized_docs)]
        documents = stem_and_remove_stop_words(tokenized_docs, stop_words, self.use_stemming, self.strip_non_alphabetical)
        return documents
        
    def tokenize_all_documents(self, raw_documents):
        stop_words = self.get_stop_words()
        preprocess = self.build_preprocessor()
        tokenized_docs = [sent_detector.tokenize(preprocess(self.decode(doc)).strip(), realign_boundaries=False) for doc in raw_documents]
        tokenized_docs = [word_tokenize(sent) for sent in itertools.chain.from_iterable(tokenized_docs)]
        documents = stem_and_remove_stop_words(tokenized_docs, stop_words, self.use_stemming, self.strip_non_alphabetical)
        return documents

    def tokenize_all_documents(self, raw_documents):
        stop_words = self.get_stop_words()
        preprocess = self.build_preprocessor()
        tokenized_docs = [sent_detector.tokenize(preprocess(self.decode(doc)).strip(), realign_boundaries=False) for doc in raw_documents]
        tokenized_docs = [word_tokenize(sent) for sent in itertools.chain.from_iterable(tokenized_docs)]
        documents = stem_and_remove_stop_words(tokenized_docs, stop_words, self.use_stemming, self.strip_non_alphabetical)
        return documents

    def tokenize_all_documents(self, raw_documents):
        stop_words = self.get_stop_words()
        preprocess = self.build_preprocessor()
        tokenized_docs = [sent_detector.tokenize(preprocess(self.decode(doc)).strip(), realign_boundaries=False) for doc in raw_documents]
        tokenized_docs = [word_tokenize(sent) for sent in itertools.chain.from_iterable(tokenized_docs)]
        documents = stem_and_remove_stop_words(tokenized_docs, stop_words, self.use_stemming, self.strip_non_alphabetical)
        return documents


    def get_ngrams_by_sent(self, sentences):
        all_ngrams = list(map(lambda s: self._word_ngrams(s), sentences))
        return reduce(operator.add, all_ngrams)

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer

        if self.analyzer == 'word':
            return lambda doc: self.get_ngrams_by_sent(self.tokenize_documents(doc))

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

    def get_stop_words(self):
        if not self.stop_words or self.stop_words == 'english':
            return super(TfidfVectorizer, self).get_stop_words()
        elif os.path.exists(self.stop_words):
            with open(self.stop_words) as f:
                return [line.rstrip() for line in f]
        else:
            print("Stop words configured as: {}".format(self.stop_words))
            print("If a file was intended, it's path could not be found. "
                  "Will continue trying to handle in case another type was intended, but stop words likely contains a path which needs to be fixed.")
            return super(TfidfVectorizer, self).get_stop_words()

    @property
    def feature_names_(self):
        return super(TfidfVectorizer, self).get_feature_names()
