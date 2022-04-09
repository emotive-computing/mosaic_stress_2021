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

def run_generalizability(base_dir):

    tokenizer = get_bert_tokenizer()

    # TODO: FIX to correct label
    sentence_column = Settings.BERT_FEATURES.sentence_column_name
    for model_run_instance in ModelRunner.get_all_model_run_instances():
        label_instance = model_run_instance.label
        data_dir = os.path.join(base_dir, 'dataset', label_instance)
        model_dir = os.path.join(base_dir, 'saved_models', label_instance)
        results_per_fold_dir = os.path.join(base_dir, 'results_per_fold', label_instance)
        results_dir = os.path.join(base_dir, 'results', label_instance)
        for experiment in Settings.COLUMNS.GENERALIZABILITY_FILTERS:
            print("\n\nTesting Generalizability for: {}".format(experiment["experiment_id"]))
            print("\n\nEvaluating models for label: {}".format(model_run_instance.label))
            X, y, le = Dataset().get(model_run_instance.label, model_run_instance.feature_source, custom_column_filters=experiment['filters'])
            X, y = shuffle(X, y, random_state=Settings.RANDOM_STATE)
            test_group_value = X.iloc[0][experiment["GROUPID_COLUMN"]]  if "GROUPID_COLUMN" in experiment else "NA"
            print("\nExperiment Group ID: {}".format(test_group_value))
            print("\nNum Test Examples: {}".format(len(y)))

            # TODO: Take these from settings file.
            instance_config = Settings.COLUMNS.BERT_LABELS[label_instance]
            BATCH_SIZE = 32
            num_folds = Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS
            num_labels = instance_config['num_labels']

            if instance_config['is_multilabel']:
                label_list = instance_config['label_list']
            else:
                label_list = [0, 1]

            overall_predictions = None

            for fold_num in range(1, num_folds+1):

                # paths for loading model for fold
                model_path = os.path.join(model_dir, str(fold_num), 'model.bin')
                fold_result_path = os.path.join(results_per_fold_dir, str(fold_num))


                # Get data for fold
                print('Collecting data for fold {}'.format(fold_num))
                X_test, y_test = X, y
                print('Num Test: ', len(y_test))
                # Convert labels to floats
                print('Parsing data for testing...')
                if instance_config['is_multilabel']:
                    y_test = X_test[label_list].apply(pd.to_numeric, errors='coerce').fillna(0).apply(lambda x: [0 if y <= 0 else 1 for y in x]).values
                    # y_test = [list(map(float, X_test.iloc[i][label_instance].split(','))) for i in range(len(X_test))]
                else:
                    # binary encode
                    if instance_config['convert_to_onehot']:
                        onehot_encoder = OneHotEncoder(sparse=False)
                        y_test = np.array(y_test)
                        y_test = y_test.reshape(len(y_test), 1)
                        y_test = onehot_encoder.fit_transform(y_test)
                        num_labels = 2


                # Loading BERT model
                print('Loading model for fold: {}'.format(fold_num))

                # Load model to make sure it works
                model = load_bert_model(model_path, num_labels=num_labels)

                # Evaluate model
                metrics, predictions = evaluate_model(model, X_test, y_test, tokenizer, label_list=label_list)

                # Write results to file
                write_results_to_disk(fold_result_path, metrics, predictions,label_list, append_to_name="_"+experiment["experiment_id"]+"_generalizability")

                #Add to Overall Predictions
                if overall_predictions is None:
                    overall_predictions = predictions
                else:
                    overall_predictions[predictions.columns[2:]] += predictions[predictions.columns[2:]]
            print('Testing Complete.')

            overall_predictions[predictions.columns[2:]] /= num_folds
            all_logits = overall_predictions[[label+'_logits' for label in label_list]].values
            all_labels = overall_predictions[[label+'_true' for label in label_list]].values
            #     ROC-AUC & AUPRC calcualation
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            auprc = dict()

            for i in range(num_labels):
                fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                auprc[i] = average_precision_score(all_labels[:, i], all_logits[:, i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            overall_metrics = {
                                'roc_auc': roc_auc,
                                'auprc': auprc

                            }


            print('Overall metrics:  {}'.format(overall_metrics))

            write_results_to_disk(results_dir, overall_metrics, overall_predictions, label_list, append_to_name="_"+experiment["experiment_id"]+"_generalizability")

def main():
    SettingsModuleLoader.init_settings()
    base_dir = Settings.IO.RESULTS_OUTPUT_FOLDER
    run_generalizability(base_dir)


if __name__ == '__main__':
    main()
