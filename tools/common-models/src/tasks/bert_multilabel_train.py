import sys
import os
import gc
import errno
import joblib
import numpy as np
import pandas as pd
import traceback

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torch
from transformers import BertTokenizer, WordpieceTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForSequenceClassification, BertConfig, BertPreTrainedModel, BertForPreTraining, BertModel, BertForMaskedLM, AdamW
from transformers import get_linear_schedule_with_warmup
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss
import torch.nn.functional as F
from tqdm import tqdm, trange
import random
from sklearn.metrics import roc_curve, auc, average_precision_score, r2_score, mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from math import sqrt

import time
import datetime
import json

from src.configuration.settings_module_loader import SettingsModuleLoader
from src.configuration.settings_template import Settings
import src.tasks.generate_splits as create_splits
from src.run.model_runner import ModelRunner
from src.models.pytorch_utils import EarlyStopping, show_gpu_usage


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

def ensure_directory(dirname):
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def ensure_directory_for_file(filename):
    directory = os.path.dirname(filename)
    ensure_directory(directory)
    return filename





def get_data_for_fold(data_dir, fold_num = 1):
    X_train = pd.read_pickle(os.path.join(data_dir, str(fold_num), 'x-train.pkl'))
    X_test = pd.read_pickle(os.path.join(data_dir, str(fold_num), 'x-test.pkl'))
    y_train = np.load(os.path.join(data_dir, str(fold_num), 'y-train.npy'))
    y_test = np.load(os.path.join(data_dir, str(fold_num), 'y-test.npy'))
    print('Collected data from {}'.format(os.path.join(data_dir, str(fold_num))))
    return X_train, y_train, X_test, y_test





def get_bert_tokenizer():
    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return tokenizer





def encode_sentences(sentences, tokenizer, max_len=128):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in tqdm(sentences):
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            max_length = max_len,          # Truncate all sentences.
                            truncation=True,
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)


    # Set the maximum sequence length.
    MAX_LEN = max_len

    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

    # print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in tqdm(input_ids):

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return input_ids, attention_masks




class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels        
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        

        if labels is not None:
            if Settings.PREDICTION.is_regression():
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            
            
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True





def create_dataloaders(X_train, y_train, tokenizer,  batch_size=32, max_len=128, validation_fraction=0.1):
    """Convert X_train(list of sentences) to BERT encodings and create pytorch

       Arguments:
           X_train, list(str): list of train sentences
           y_train, list(float): list of labels per sentence for multilabel
           batch_size, int: number of examples per batch. Bert recommends 32
           tokenizer, BertTokenizer: Loaded BertTokenizer object
           validation_fraction, float (optional): Percentage to split by for validation.
    """
    train_inputs, train_masks  = encode_sentences(X_train, tokenizer, max_len=max_len)
    # Split training data for validation
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_inputs, y_train,
                                                                random_state=Settings.RANDOM_STATE, test_size=validation_fraction)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(train_masks, y_train,
                                                 random_state=Settings.RANDOM_STATE, test_size=validation_fraction)

    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels, dtype=torch.float)
    validation_labels = torch.tensor(validation_labels, dtype=torch.float)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # The DataLoader needs to know our batch size for training, so we specify it
    # here.
    # For fine-tuning BERT on a specific task, the authors recommend a batch size of
    # 16 or 32.

    batch_size = batch_size

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return train_dataloader, validation_dataloader





def get_model(num_labels=2, use_cuda=False):
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForMultiLabelSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels, # The number of output labels--2 for binary classification.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    if use_cuda:
        # Tell pytorch to run this model on the GPU.
        model.cuda()
    return model





def print_model_params(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))





# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
#     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
    return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()

def r_squared(y_pred:Tensor, y_true:Tensor):
    "Compute r2_score when `y_pred` and `y_true` are the same size."
    yTrue = y_true.detach().cpu().numpy()
    predictions = y_pred.detach().cpu().numpy()
    return r2_score(yTrue, predictions)


def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean().item()

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))





def create_train_config(train_dataloader,
                        validation_dataloader,
                        num_labels,
                        epochs=2,
                        adam_lr=2e-5,
                        adam_eps=1e-8,
                        num_warmup_steps=0,
                        random_seed=42):

    return {
        'train_dataloader': train_dataloader,
        'validation_dataloader': validation_dataloader,
        'num_labels': num_labels,
        'epochs': epochs,
        'adam_lr': adam_lr,
        'adam_eps': adam_eps,
        'num_warmup_steps': num_warmup_steps,
        'random_seed': random_seed,
    }


def train_bert_model(config, model_path=''):
    """Trains a BERT model for multilabel sequence classification

    Arguments:
        config, dict: Dictionary config with following keys:
            {
                'train_dataloader': Pytorch data loader for train data of the form (input_ids, input_mask, labels))
                'validation_dataloader': Pytorch data loader for validation data
                'num_labels': Number of labels
                'epochs':  Number of training epochs,
                'adam_lr': Learning rate for Adam Optimizer,
                'adam_eps': Epsilon for Adam optimizer,
                'num_warmup_steps': Optimizer warmup steps - default is 0,
                'random_seed': Seed for reproducibility
            }
        model_path, string: path to save model binary - for use with early stopping.
    Returns model: The trained Pytorch Model
    """
    if ngpu > 0:
        model = get_model(use_cuda=True, num_labels=config['num_labels'])
    else:
        model = get_model(use_cuda=False, num_labels=config['num_labels'])

    train_dataloader = config['train_dataloader']
    validation_dataloader = config['validation_dataloader']

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr = config['adam_lr'], # args.learning_rate - default is 5e-5, using 2e-5
                      eps = config['adam_eps'] # args.adam_epsilon  - default is 1e-8.
                    )


    # Number of training epochs (authors recommend between 2 and 4)
    epochs = config['epochs']

    num_labels = config['num_labels']

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = config['num_warmup_steps'], # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


    # Set the seed value all over the place to make this reproducible.
    seed_val = config['random_seed']

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    early_stopping = None
    hyper_params = Settings.CROSS_VALIDATION.HYPER_PARAMS.BERT
    if 'early_stopping_patience' in hyper_params:
        early_stopping = EarlyStopping(hyper_params['early_stopping_patience'], verbose=True, path=model_path)

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(tqdm(train_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            try:
                loss = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            except Exception as e:
                del model
                clear_gpu_memory()
                raise e

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits = None
        all_labels = None

        # Evaluate data for one epoch
        for batch in tqdm(validation_dataloader):

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs

            # Move logits and labels to CPU
            logits = logits.detach().cpu()
            label_ids = b_labels.to('cpu')
            if Settings.PREDICTION.is_regression():
               tmp_eval_accuracy = r_squared(logits, label_ids)
            else:
                tmp_eval_accuracy = accuracy_thresh(logits, label_ids)

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

            if all_labels is None:
                all_labels = label_ids.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)


            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        
        if early_stopping is not None:
             to_save_model = early_stopping(eval_loss)
             if to_save_model:
                 save_bert_model(model, model_path)
             if early_stopping.early_stop:
                print("Early stopping")
                break

        if Settings.PREDICTION.is_classification():
            #     ROC-AUC calcualation
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(num_labels):
                fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            result = {'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
                    'roc_auc': roc_auc  }
            print(result)
        elif Settings.PREDICTION.is_regression():
            # modify this to support multiple outputs
            
            pearson_score = pearsonr(all_labels[:], all_logits[:, 0])[0]
            spearman_score = spearmanr(all_labels[:], all_logits[:, 0])[0]
            result = {
                'eval_loss': eval_loss,
                'eval_r2_score': eval_accuracy,
                'pearsonr': pearson_score,
                'spearmanr': spearman_score,
            }
            print(result)
    if early_stopping is not None:
        del model, optimizer, scheduler, train_dataloader, validation_dataloader
        clear_gpu_memory()
        model = load_bert_model(os.path.join(model_path, "model.bin"), num_labels)
    
    print("Training complete!")
    return model


def clear_gpu_memory():
    if ngpu > 0:
        gc.collect()
        torch.cuda.empty_cache()
        show_gpu_usage("GPU Usage after clearing cache")


def save_bert_model(model, model_dir):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(model_dir, "model.bin")
    ensure_directory_for_file(output_model_file)
    torch.save(model_to_save.state_dict(), output_model_file)
    print('Model saved to file: {}'.format(output_model_file))
    return output_model_file

def load_bert_model(model_path, num_labels=2):
    # Load a trained model that you have fine-tuned
    print('Loading model from file: {}'.format(model_path))
    if torch.cuda.is_available():
        model_state_dict = torch.load(model_path)
    else:
        model_state_dict = torch.load(model_path, map_location='cpu')
    model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased", num_labels = num_labels, state_dict=model_state_dict)
    if ngpu > 0:
        model.to(device)
    return model



def predict_from_model(model, X_test, tokenizer, label_list=None, batch_size=32, max_seq_len=128, data_loader=None):

    sentence_column = Settings.BERT_FEATURES.sentence_column_name
    ID_column = Settings.COLUMNS.IDENTIFIER

    # Hold input data for returning it
    input_data = [{ 'id': input_example[ID_column], 'sentence': input_example[sentence_column] } for _, input_example in X_test.iterrows()]
    if data_loader is None:
        test_sentences = X_test[sentence_column]
        test_inputs, test_masks  = encode_sentences(test_sentences, tokenizer, max_len=max_seq_len)
        # Convert all inputs and labels into torch tensors, the required datatype
        # for our model.
        test_inputs = torch.tensor(test_inputs)
        test_masks = torch.tensor(test_masks)

        # Create the DataLoader for testset
        test_data = TensorDataset(test_inputs, test_masks)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    else:
        test_dataloader = data_loader

    all_logits = None
    all_probabilities = None
    all_binary_predictions = None


    # Evaluate data for one epoch
    for batch in tqdm(test_dataloader):

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        if sys.platform.startswith('win'):
            b_input_ids = b_input_ids.type(torch.LongTensor)
            b_input_mask = b_input_mask.type(torch.LongTensor)

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs

        # Move logits and labels to CPU
        logits = logits.detach().cpu()


        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        if all_probabilities is None:
            all_probabilities = logits.detach().cpu().sigmoid().numpy()
        else:
            all_probabilities = np.concatenate((all_probabilities, logits.detach().cpu().sigmoid().numpy()), axis=0)

        if Settings.PREDICTION.is_classification():
            if all_binary_predictions is None:
                all_binary_predictions = (logits.detach().cpu().sigmoid()>0.5).int().numpy()
            else:
                all_binary_predictions = np.concatenate((all_binary_predictions, (logits.detach().cpu().sigmoid()>0.5).int().numpy()), axis=0)


    logit_labels = [label+'_logits' for label in label_list]
    predictions = pd.merge(pd.DataFrame(input_data), pd.DataFrame(all_logits, columns=logit_labels), left_index=True, right_index=True)
    if Settings.PREDICTION.is_classification():
        prob_labels = [label+'_probs' for label in label_list]
        pred_bin_labels = [label+'_predicted' for label in label_list]
        predictions = pd.merge(predictions, pd.DataFrame(all_probabilities, columns=prob_labels), left_index=True, right_index=True)
        predictions = pd.merge(predictions, pd.DataFrame(all_binary_predictions, columns=pred_bin_labels), left_index=True, right_index=True)
    return predictions, test_dataloader


def evaluate_model(model, X_test, y_test, tokenizer, label_list=None, batch_size=32, max_seq_len=128):

    sentence_column = Settings.BERT_FEATURES.sentence_column_name
    ID_column = Settings.COLUMNS.IDENTIFIER
    # Hold input data for returning it
    input_data = [{ 'id': input_example[ID_column], 'sentence': input_example[sentence_column] } for _, input_example in X_test.iterrows()]

    test_sentences = X_test[sentence_column]
    test_labels = y_test
    test_inputs, test_masks  = encode_sentences(test_sentences, tokenizer, max_len=max_seq_len)
    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
    test_inputs = torch.tensor(test_inputs)
    test_labels = torch.tensor(test_labels, dtype=torch.float)
    test_masks = torch.tensor(test_masks)

    # Create the DataLoader for testset
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_logits = None
    all_probabilities = None
    all_binary_predictions = None
    all_labels = None

    # Evaluate data for one epoch
    for batch in tqdm(test_dataloader):

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids,
                         token_type_ids=None,
                         attention_mask=b_input_mask,
                         labels=b_labels)

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs

        # Move logits and labels to CPU
        logits = logits.detach().cpu()
        label_ids = b_labels.to('cpu')

        if Settings.PREDICTION.is_regression():
            tmp_eval_accuracy = r_squared(logits, label_ids)
        else:
            tmp_eval_accuracy = accuracy_thresh(logits, label_ids)

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        if all_probabilities is None:
            all_probabilities = logits.detach().cpu().sigmoid().numpy()
        else:
            all_probabilities = np.concatenate((all_probabilities, logits.detach().cpu().sigmoid().numpy()), axis=0)

        if Settings.PREDICTION.is_classification():
            if all_binary_predictions is None:
                all_binary_predictions = (logits.detach().cpu().sigmoid()>0.5).int().numpy()
            else:
                all_binary_predictions = np.concatenate((all_binary_predictions, (logits.detach().cpu().sigmoid()>0.5).int().numpy()), axis=0)


        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)


        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    
    if Settings.PREDICTION.is_classification():
        #     ROC-AUC calcualation
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        auprc = dict()

        num_labels = len(label_list)
        for i in range(num_labels):
            fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            auprc[i] = average_precision_score(all_labels[:, i], all_logits[:, i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        metrics = {
                    'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
    #               'loss': tr_loss/nb_tr_steps,
                    'roc_auc': roc_auc,
                    'auprc': auprc,
                }
        print(metrics)
    elif Settings.PREDICTION.is_regression():
        # modify this to support multiple outputs            
        pearson_score = pearsonr(all_labels[:], all_logits[:, 0])[0]
        spearman_score = spearmanr(all_labels[:], all_logits[:, 0])[0]
        metrics = {
            'eval_loss': eval_loss,
            'eval_r2_score': eval_accuracy,
            'pearsonr': pearson_score,
            'spearmanr': spearman_score,
        }
        print(metrics)
    true_labels = [label+'_true' for label in label_list]
    logit_labels = [label+'_logits' for label in label_list]
    predictions = pd.merge(pd.DataFrame(input_data), pd.DataFrame(all_logits, columns=logit_labels), left_index=True, right_index=True)
    if Settings.PREDICTION.is_classification():
        prob_labels = [label+'_probs' for label in label_list]
        pred_bin_labels = [label+'_predicted' for label in label_list]        
        predictions = pd.merge(predictions, pd.DataFrame(all_probabilities, columns=prob_labels), left_index=True, right_index=True)
        predictions = pd.merge(predictions, pd.DataFrame(all_binary_predictions, columns=pred_bin_labels), left_index=True, right_index=True)
    predictions = pd.merge(predictions, pd.DataFrame(all_labels, columns=true_labels), left_index=True, right_index=True)
    return metrics, predictions


def write_predictions_to_disk(result_dir, predictions, label_list, append_to_name=''):
        predictions_file_name = os.path.join(result_dir,'predictions'+append_to_name+'.csv')
        ensure_directory_for_file(predictions_file_name)
        predictions.to_csv(predictions_file_name)
        print('Wrote {}'.format(predictions_file_name))
        return predictions_file_name



def write_results_to_disk(result_dir, metrics, predictions, label_list, append_to_name=''):
        predictions_file_name = os.path.join(result_dir,'predictions'+append_to_name+'.csv')
        metrics_file_name = os.path.join(result_dir,'metrics'+append_to_name+'.csv')
        ensure_directory_for_file(predictions_file_name)
        ensure_directory_for_file(metrics_file_name)
        predictions.to_csv(predictions_file_name)
        if Settings.PREDICTION.is_classification():
            metrics_csv = {
                'roc_auc_micro': metrics['roc_auc']['micro']
            }
            for i in range(len(label_list)):
                metrics_csv['roc_auc_{}'.format(label_list[i])] = metrics['roc_auc'][i]
                metrics_csv['auprc_{}'.format(label_list[i])] = metrics['auprc'][i]
            metrics_df = pd.DataFrame([metrics_csv])
            metrics_df.to_csv(metrics_file_name)
        elif Settings.PREDICTION.is_regression():
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(metrics_file_name)
        print('Wrote {}, {}'.format(predictions_file_name, metrics_file_name))
        return predictions_file_name, metrics_file_name


def run_bert_training(base_dir):

    tokenizer = get_bert_tokenizer()
    sentence_column = Settings.BERT_FEATURES.sentence_column_name
    for model_run_instance in ModelRunner.get_all_model_run_instances():
        label_instance = model_run_instance.label
        print('Training model for label: {}'.format(label_instance))
        data_dir = os.path.join(base_dir, 'dataset', label_instance)
        model_dir = os.path.join(base_dir, 'saved_models', label_instance)
        results_per_fold_dir = os.path.join(base_dir, 'results_per_fold', label_instance)
        results_dir = os.path.join(base_dir, 'results', label_instance)

        # TODO: Take these from settings file.
        instance_config = Settings.COLUMNS.BERT_LABELS[label_instance]
        BATCH_SIZE = Settings.CROSS_VALIDATION.HYPER_PARAMS.BERT['batch_size'] if 'batch_size' in Settings.CROSS_VALIDATION.HYPER_PARAMS.BERT else 32
        MAX_LEN = Settings.CROSS_VALIDATION.HYPER_PARAMS.BERT['max_seq_len'] if 'max_seq_len' in Settings.CROSS_VALIDATION.HYPER_PARAMS.BERT else 128
        num_folds = Settings.CROSS_VALIDATION.NUM_TRAIN_TEST_FOLDS
        num_labels = instance_config['num_labels']
        if instance_config['is_multilabel']:
            label_list = instance_config['label_list']
        else:
            if instance_config['convert_to_onehot']:
                label_list = ["0", "1"]
            else:
                label_list = [label_instance]


        overall_predictions = None

        for fold_num in range(1, num_folds+1):

            # paths for saving model and results of this fold
            model_path = os.path.join(model_dir, str(fold_num))
            fold_result_path = os.path.join(results_per_fold_dir, str(fold_num))


            # Get data for fold
            print('Collecting data for fold {}'.format(fold_num))
            X_train, y_train, X_test, y_test = get_data_for_fold(data_dir, fold_num)
            print('Num Train: ', len(y_train))
            print('Num Test: ', len(y_test))
            # Convert labels to floats
            print('Parsing data for training...')
            if instance_config['is_multilabel']:
                try:
                    y_train = X_train[label_list].apply(pd.to_numeric, errors='coerce').fillna(0).apply(lambda x: [0 if y <= 0 else 1 for y in x]).values
                    y_test = X_test[label_list].apply(pd.to_numeric, errors='coerce').fillna(0).apply(lambda x: [0 if y <= 0 else 1 for y in x]).values
                    # y_train = [list(map(float, X_train.iloc[i][label_instance].split(','))) for i in range(len(X_train))]
                    # y_test = [list(map(float, X_test.iloc[i][label_instance]. split(','))) for i in range(len(X_test))]
                except:
                    raise Exception('Label encoding failed')
            else:
                # binary encode
                if instance_config['convert_to_onehot']:
                    onehot_encoder = OneHotEncoder(sparse=False)
                    y_train = np.array(y_train)
                    y_train = y_train.reshape(len(y_train), 1)
                    y_train = onehot_encoder.fit_transform(y_train)
                    y_test = np.array(y_test)
                    y_test = y_test.reshape(len(y_test), 1)
                    y_test = onehot_encoder.fit_transform(y_test)
                    num_labels = 2

            train_sentences = X_train[sentence_column]
            train_labels = y_train
            train_dataloader, validation_dataloader = create_dataloaders(train_sentences, train_labels, tokenizer,
                                                                         batch_size=BATCH_SIZE, max_len=MAX_LEN, validation_fraction=0.1)

            # Initialize training config
            epochs = Settings.CROSS_VALIDATION.HYPER_PARAMS.BERT['epochs'] if 'epochs' in Settings.CROSS_VALIDATION.HYPER_PARAMS.BERT else 2
            train_config = create_train_config(train_dataloader,
                                               validation_dataloader,
                                               num_labels=num_labels,
                                               epochs=epochs,
                                               random_seed=Settings.RANDOM_STATE)


            # Train BERT model
            model = None
            if not Settings.LOAD_MODELS:
                print('Training model for fold: {}'.format(fold_num))
                retries = 0
                while retries < 2:
                    try:
                        model = train_bert_model(train_config, model_path)
                        break
                    except Exception as e:
                        traceback.print_exc()
                        clear_gpu_memory()                        
                        retries += 1
                        print("Retrying fold.")
                # Save Model
                if model is None:
                    raise Exception("Failed to train model.")
                model_file = save_bert_model(model, model_path)
            else:
                model_file =  os.path.join(model_path, "model.bin")
                print('Loading saved model for fold: {}'.format(fold_num))

            del model
            clear_gpu_memory()
            # Load model to make sure it works
            model = load_bert_model(model_file, num_labels=num_labels)

            # Evaluate model
            metrics, predictions = evaluate_model(model, X_test, y_test, tokenizer, label_list=label_list, batch_size=BATCH_SIZE, max_seq_len=MAX_LEN)

            # Write results to file
            write_results_to_disk(fold_result_path, metrics, predictions, label_list=label_list)

            #Add to Overall Predictions
            if overall_predictions is None:
                overall_predictions = predictions
            else:
                overall_predictions = pd.concat([overall_predictions, predictions])
            
            del model
            clear_gpu_memory()
        print('Training Complete.')

        
        all_logits = overall_predictions[[label+'_logits' for label in label_list]].values
        all_labels = overall_predictions[[label+'_true' for label in label_list]].values
        
        if Settings.PREDICTION.is_regression():
            # modify this to support multiple outputs            
            pearson_score = pearsonr(all_labels[:, 0], all_logits[:, 0])[0]
            spearman_score = spearmanr(all_labels[:, 0], all_logits[:, 0])[0]
            overall_metrics = {
                'rmse': sqrt(mean_squared_error(all_labels[:, 0], all_logits[:, 0])),
                'r2_score': r2_score(all_labels[:, 0], all_logits[:, 0]),
                'pearsonr': pearson_score,
                'spearmanr': spearman_score,
            }
            print('Overall metrics:  {}'.format(overall_metrics))
            write_results_to_disk(results_dir, overall_metrics, overall_predictions, label_list=label_list)
        elif Settings.PREDICTION.is_classification():
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

            write_results_to_disk(results_dir, overall_metrics, overall_predictions, label_list=label_list)

def main():
    create_splits.main()
    base_dir = Settings.IO.RESULTS_OUTPUT_FOLDER
    run_bert_training(base_dir)


if __name__ == '__main__':
    main()
