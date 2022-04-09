import os
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from commonmodels2.stages.pipeline import Pipeline
from commonmodels2.models.model import TensorFlowModel
from commonmodels2.stages.load_data import DataFrameLoaderStage, CSVLoaderStage
from commonmodels2.stages.preprocessing import FeatureScalerPreprocessingStage
from commonmodels2.stages.cross_validation import GridParamSearch, GenerateCVFoldsStage, CrossValidationStage, SupervisedCVContext, TensorFlowModelTuningContext
from commonmodels2.stages.training_stage import SupervisedTrainingContext
from commonmodels2.stages.evaluation_stage import SupervisedEvaluationContext
from commonmodels2.utils.utils import get_tensorflow_loss_func, get_tensorflow_metric_func, flatten_df_cols, one_hot_encode_cols

def my_tf_model_func(params):
    cnn_model = Sequential()
    cnn_model.add(Reshape((28,28,1), input_shape=(784,)))
    cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu', padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu', padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.2))

    cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu', padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu', padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.2))

    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=params['dense_layer_size'], activation='relu'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Dense(units=10, activation='softmax'))
    return cnn_model

def RunML():

    if not os.path.isfile('fashion_mnist.csv'):
        # Download the Fashion MNIST dataset
        (mnist_train, mnist_test), mnist_info = tfds.load('fashion_mnist', split=['train', 'test'],
                                                          shuffle_files=True, as_supervised=True,
                                                          with_info=True)
        mnist_train_df = tfds.as_dataframe(mnist_train, mnist_info)
        mnist_test_df = tfds.as_dataframe(mnist_test, mnist_info)
        mnist_df = pd.concat((mnist_train_df, mnist_test_df), axis=0)
        mnist_df.reset_index(inplace=True, drop=True)
        (mnist_df, feat_cols) = flatten_df_cols(mnist_df, ['image'])
        (mnist_df, label_cols) = one_hot_encode_cols(mnist_df, ['label'])
        mnist_df.to_csv('fashion_mnist.csv', index= False, header= True)
        feat_cols = pd.DataFrame(data=feat_cols, columns=['features'])
        feat_cols.to_csv('fashion_mnist_feature_names.csv', index = False, header = False)
        label_cols = pd.DataFrame(data=label_cols, columns=['labels'])
        label_cols.to_csv('fashion_mnist_label_names.csv', index=False, header=False)
    else:
        mnist_df = pd.read_csv('fashion_mnist.csv')
        feat_cols = pd.read_csv('fashion_mnist_feature_names.csv', header=None)
        label_cols = pd.read_csv('fashion_mnist_label_names.csv', header=None)

    p = Pipeline()

    # Load Fashion MNIST data into the pipeline
    s0 = CSVLoaderStage()
    s0.setFilePath('fashion_mnist.csv')
    p.addStage(s0)

    # Use the provided train/test split from the Fashion MNIST dataset
    s1 = GenerateCVFoldsStage(strategy='random',
                              strategy_args={'num_folds': 5,
                                             'seed': 42})
    p.addStage(s1)

    s2 = FeatureScalerPreprocessingStage(feat_cols, 'min-max', feature_range=(0.0, 1.0))
    s2.set_fit_transform_data_idx(range(mnist_df.shape[0]))
    p.addStage(s2)

    # Cross-validation for hyperparameter tuning
    s3 = CrossValidationStage()
    cv_context = SupervisedCVContext()
    training_context = SupervisedTrainingContext()
    tfm = TensorFlowModel()
    tfm.set_model_create_func(my_tf_model_func)
    tfm.set_fit_params({'epochs': 10, 'batch_size': 512})
    tfm.set_compile_params({'optimizer': 'adam', 'loss': 'CategoricalCrossentropy', 'metrics': 'accuracy'})

    tfm.set_model_params({'dense_layer_size': 128})
    training_context.model = tfm
    training_context.feature_cols = feat_cols
    training_context.label_cols = label_cols
    cv_context.training_context = training_context

    eval_context = SupervisedEvaluationContext()
    eval_context.label_cols = label_cols
    eval_context.eval_funcs = get_tensorflow_metric_func('CategoricalAccuracy')
    cv_context.eval_context = eval_context

    s3.setCVContext(cv_context)
    p.addStage(s3)

    p.run()

    p.getDC.save('tf_fashion_results')

RunML()
