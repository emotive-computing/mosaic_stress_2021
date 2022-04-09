import os
import errno
import torch
import random
from sklearn.externals import joblib
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

from src.configuration.settings_module_loader import SettingsModuleLoader
from src.configuration.settings_template import Settings
from src.io.read_data_input import Dataset
from src.run.model_runner import ModelRunner
from src.tasks.bert_multilabel_train import *




# If there's a GPU available...
ngpu = 0
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    ngpu = torch.cuda.device_count()
    print('There are %d GPU(s) available.' % ngpu)

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if ngpu > 0:
    torch.cuda.manual_seed_all(42)

def run_predictions(base_dir):

    tokenizer = get_bert_tokenizer()
    sentence_column = Settings.BERT_FEATURES.sentence_column_name
    data_loader = None
    for model_run_instance in ModelRunner.get_all_model_run_instances():
        label_instance = model_run_instance.label
        model_dir = os.path.join(Settings.IO.BERT_MODEL_DIR, label_instance)
        results_per_fold_dir = os.path.join(base_dir, 'results_per_fold', label_instance)
        results_dir = os.path.join(base_dir, 'results', label_instance)
        print("\nPredicting for label: {}".format(model_run_instance.label))
        X, y, le = Dataset().get(model_run_instance.label, model_run_instance.feature_source, custom_column_filters=Settings.COLUMNS.BERT_PREDICT_FILTERS)
        X, y = shuffle(X, y, random_state=Settings.RANDOM_STATE)
        print("\nNum Test Examples: {}".format(len(X)))

        instance_config = Settings.COLUMNS.BERT_LABELS[label_instance]
        BATCH_SIZE = Settings.CROSS_VALIDATION.HYPER_PARAMS.BERT['batch_size'] if \
            'batch_size' in Settings.CROSS_VALIDATION.HYPER_PARAMS.BERT else 32
        MAX_LEN = Settings.CROSS_VALIDATION.HYPER_PARAMS.BERT['max_seq_len'] if 'max_seq_len' in Settings.CROSS_VALIDATION.HYPER_PARAMS.BERT else 128
        num_folds = Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS
        num_labels = instance_config['num_labels']

        if instance_config['is_multilabel']:
            label_list = instance_config['label_list']
        else:
            label_list = ["0", "1"]
            num_labels = 2

        overall_predictions = None

        for fold_num in range(1, num_folds+1):

            # paths for loading model for fold
            model_path = os.path.join(model_dir, str(fold_num), 'model.bin')
            fold_result_path = os.path.join(results_per_fold_dir, str(fold_num))


            # Get data for fold
            print('Collecting data for fold {}'.format(fold_num))
            X_test, y_test = X, y
            print('Num Test: ', len(y_test))


            # Loading BERT model
            print('Loading model for fold: {}'.format(fold_num))

            # Load model to make sure it works
            model = load_bert_model(model_path, num_labels=num_labels)

            # Convert labels to floats
            print('Parsing data for predictions...')

            # Evaluate model
            predictions, data_loader = predict_from_model(model, X_test, tokenizer, label_list=label_list, batch_size=BATCH_SIZE, max_seq_len=MAX_LEN, data_loader=data_loader)

            # Write results to file
            write_predictions_to_disk(fold_result_path, predictions,label_list)

            #Add to Overall Predictions
            if overall_predictions is None:
                overall_predictions = predictions
            else:
                overall_predictions[predictions.columns[2:]] += predictions[predictions.columns[2:]]
        print('Testing Complete.')

        overall_predictions[predictions.columns[2:]] /= num_folds
        write_predictions_to_disk(results_dir, overall_predictions, label_list)

def main():
    SettingsModuleLoader.init_settings()
    base_dir = Settings.IO.RESULTS_OUTPUT_FOLDER
    run_predictions(base_dir)


if __name__ == '__main__':
    main()
