import sys, os

# Import everything from keras with this wrapper to prevent printouts of "Using TensorFlow backend."
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
sys.stderr = stderr
import tensorflow as tf
import numpy as np
import sklearn.metrics as skm
MASK_VALUE = -1

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value



def _mask_nan(y_true, y_pred):
    mask_array = ~np.isnan(y_true)
    if np.any(np.isnan(y_pred)):
        print("WARNING: y_pred contains {0}/{1} np.nan values. removing them...".
              format(np.sum(np.isnan(y_pred)), y_pred.size))
        mask_array = np.logical_and(mask_array, ~np.isnan(y_pred))
    return y_true[mask_array], y_pred[mask_array]

def _mask_value(y_true, y_pred, mask=MASK_VALUE):
    mask_array = y_true != mask
    return y_true[mask_array], y_pred[mask_array]


def _mask_value_nan(y_true, y_pred, mask=MASK_VALUE):
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return _mask_value(y_true, y_pred, mask)

def auprc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred, curve='PR')

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auprc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value