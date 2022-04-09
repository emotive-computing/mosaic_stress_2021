# Holds settings values from the settings file passed in using a global variable title config
import os

from src.common.descriptors import TypeDescriptorMeta, ListTypeDescriptor, TypeDescriptor, AllowNoneTypeDescriptor
from src.configuration.settings_enum_options import SettingsEnumOptions

class Settings(metaclass=TypeDescriptorMeta):

    FEATURE_INPUT_SOURCES_TO_RUN = ListTypeDescriptor(type(SettingsEnumOptions.FeatureInput))
    PREDICTION = TypeDescriptor(SettingsEnumOptions.Prediction)

    class BERT_FEATURES(metaclass=TypeDescriptorMeta):
        sentence_column_name = TypeDescriptor(str)

    class CROSS_VALIDATION(metaclass=TypeDescriptorMeta):

        NUM_TRAIN_TEST_FOLDS = TypeDescriptor(int)
        NUM_CV_TRAIN_VAL_FOLDS = TypeDescriptor(int)

        SCORING_FUNCTION = TypeDescriptor(object, is_optional=True) #CustomScorer

        GROUP_BY_COLUMN = ListTypeDescriptor(str, is_optional=True)
        STRATIFIED_SPLIT_CLASSES = ListTypeDescriptor(str, is_optional=True)
        STRATIFIED_BINNING_EDGES = ListTypeDescriptor(float, is_optional=True)
        STRATIFIED_NUMBER_PERCENTILE_BINS = TypeDescriptor(int, is_optional=True, default_value=10)
        SHUFFLE = TypeDescriptor(bool, is_optional=True, default_value=True)

        # USE_PRESELECTED_FOLDS = TypeDescriptor(bool, is_optional=True)
        # PRESELECTED_FOLDS_TRAIN_FILE= TypeDescriptor(bool, is_optional=True)
        # PRESELECTED_FOLDS_TEST_FILE= TypeDescriptor(bool, is_optional=True)

        class HYPER_PARAMS(metaclass=TypeDescriptorMeta):
            MODEL = TypeDescriptor(dict, is_optional=True)
            BERT = TypeDescriptor(dict, is_optional=True)
            VECTORIZER = TypeDescriptor(dict, is_optional=True)
            FEATURE_SELECTION = ListTypeDescriptor(tuple, is_optional=True)
            RESAMPLER = ListTypeDescriptor(tuple, is_optional=True)
            IMPUTER = ListTypeDescriptor(tuple, is_optional=True)
            FEATURE_SCALER = ListTypeDescriptor(tuple, is_optional=False)

    # Defines how input csv/xlsx file is set up
    class COLUMNS(metaclass=TypeDescriptorMeta):
        IDENTIFIER = TypeDescriptor(str)
        Y_LABELS_TO_PREDICT = ListTypeDescriptor(str, is_optional=True)

        GROUP_BY_COLUMN = TypeDescriptor(str, is_optional=True)

        ORDER_IN_GROUPS_BY_COLUMN = TypeDescriptor(str, is_optional=True)
        ORDER_IN_GROUPS_SORT_BY_COLUMN = TypeDescriptor(str, is_optional=True)

        COLUMN_FILTERS = TypeDescriptor(dict, is_optional=True)
        BERT_LABELS = TypeDescriptor(dict, is_optional=True)
        BERT_PREDICT_FILTERS = TypeDescriptor(dict, is_optional=True)
        GENERALIZABILITY_FILTERS = TypeDescriptor(list, is_optional=True)
        LABEL_DECLARATIONS = TypeDescriptor(dict, is_optional=True)
        MAKE_ALL_LABELS_BINARY = TypeDescriptor(bool, is_optional=True)

        GROUP_NORM_COLUMN = TypeDescriptor(str, is_optional=True, default_value=None)

        class USE_HIERARCHICAL(metaclass=TypeDescriptorMeta):
            SCHEMA = TypeDescriptor(dict, is_optional=True)
            START_NODE = TypeDescriptor(str, is_optional=True)


    USE_ONE_VS_ALL_CLF_FOR_MULTICLASS = TypeDescriptor(bool, is_optional=True)

    class IO(metaclass=TypeDescriptorMeta):
        EXCLUDE_LANGUAGE_FEATURES_FILE = TypeDescriptor(str)
        DATA_INPUT_FILE = TypeDescriptor(str)
        RESULTS_OUTPUT_FOLDER = TypeDescriptor(str)
        DATA_DIR = TypeDescriptor(str, deprecated=True)
        SAVED_MODEL_DIR = TypeDescriptor(str, is_optional=True)
        BERT_MODEL_DIR = TypeDescriptor(str, is_optional=True)
        USE_PREDEFINED_FOLDS_FILE = TypeDescriptor(str, is_optional=True, deprecated=True) # Location of predefined splits file. If does not exist, it is created if a file name is specified.
        USE_PREDEFINED_FOLDS_SPECIFIER = TypeDescriptor(str, is_optional=True)
        USE_NEW_PREDEFINED_FOLDS_FILE = TypeDescriptor(str, is_optional=True)

    MODELS_TO_RUN = ListTypeDescriptor(object)
    SAVE_FULL_DATA_FRAME = TypeDescriptor(bool, is_optional=True, default_value=False)
    LOAD_MODELS = TypeDescriptor(bool, is_optional=True)
    SAVE_MODELS = TypeDescriptor(bool, is_optional=True)
    SAVE_FEATURE_WEIGHTS = TypeDescriptor(bool, is_optional=True, default_value=False)
    RUN_FROM_SAVED_MODELS = TypeDescriptor(bool, is_optional=True)

    TASKS_TO_RUN = ListTypeDescriptor(SettingsEnumOptions.Tasks, default_value=[SettingsEnumOptions.Tasks.MODEL_COMPARISON])

    class WORD_EMBEDDINGS(metaclass=TypeDescriptorMeta):
        MAX_NUM_WORDS_IN_SEQUENCE = TypeDescriptor(int, default_value=12)
        MAX_NUM_WORDS_IN_VOCABULARY = TypeDescriptor(int, default_value=40000)
        EMBEDDING_VECTOR_SIZE = TypeDescriptor(int, default_value=100)

        class GLOVE(metaclass=TypeDescriptorMeta):
            WORD_VECTORS_ARE_TRAINABLE = TypeDescriptor(bool, default_value=True)
            USE_PRETRAINED_WORD_VECTORS_FROM_FILE = AllowNoneTypeDescriptor(str, default_value=os.path.join(os.getcwd(), "include", "glove.6B.100d.txt")) # value might be os.path.join(os.getcwd(), "include", "glove.6B.100d.txt")

    SHOW_GROUP_BY_COLUMN_VALUE = TypeDescriptor(bool, is_optional=False)
    RANDOM_STATE = TypeDescriptor(int, default_value=42)
