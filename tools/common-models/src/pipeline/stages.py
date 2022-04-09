from enum import auto

from src.common.meta import CommonEnumMeta


class PipelineStages(CommonEnumMeta):
    EXTRACT_FEATURES = auto()
    FEATURE_SCALER = auto()
    IMPUTER = auto()
    RESAMPLER = auto()
    MODEL = auto()
    DENSE = auto()
    FEATURE_SELECTION = auto()


    def get_prefix(self):
        return str(self) + "__"

class FeatureExtractionStages(CommonEnumMeta):
    SELECT = auto()
    FEATURES_FROM_DATA = auto ()
    LANGUAGE = auto()
    VECTORIZER = auto()
    WORD_EMBEDDINGS = auto()
    SEQUENCE = auto()
    INCREMENTAL_PREDICTIONS = auto()






