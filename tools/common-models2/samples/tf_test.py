import os
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.optimizers import SGD

from commonmodels2.stages.pipeline import Pipeline
from commonmodels2.models.model import TensorFlowModel
from commonmodels2.stages.load_data import DataFrameLoaderStage
from commonmodels2.stages.preprocessing import FeatureScalerPreprocessingStage
from commonmodels2.stages.cross_validation import GridParamSearch, GenerateCVFoldsStage, CrossValidationStage, SupervisedCVContext, TensorFlowModelTuningContext
from commonmodels2.stages.training_stage import SupervisedTrainingContext
from commonmodels2.stages.evaluation_stage import SupervisedEvaluationContext
from commonmodels2.utils.utils import get_tensorflow_loss_func, get_tensorflow_metric_func, flatten_df_cols, one_hot_encode_cols

def my_tf_model_func(params):
    n = params['hidden_layer_size']
    model = keras.models.Sequential()
    model.add(keras.layers.Reshape((28,28,1), input_shape=(784,)))
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(28,28,1)))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(n, activation="relu", kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def RunML():
    # Download the MNIST dataset
    (mnist_train, mnist_test), mnist_info = tfds.load('mnist', split=['train', 'test'],
                                                      shuffle_files=True, as_supervised=True,
                                                      with_info=True)
    mnist_train_df = tfds.as_dataframe(mnist_train, mnist_info)
    mnist_test_df = tfds.as_dataframe(mnist_test, mnist_info)
    mnist_df = pd.concat((mnist_train_df, mnist_test_df), axis=0)
    mnist_df.reset_index(inplace=True, drop=True)
    (mnist_df, feat_cols) = flatten_df_cols(mnist_df, ['image'])
    (mnist_df, label_cols) = one_hot_encode_cols(mnist_df, ['label'])

    p = Pipeline()

    # Load MNIST data into the pipeline
    s0 = DataFrameLoaderStage()
    s0.setDataFrame(mnist_df)
    p.addStage(s0)

    # Use the provided train/test split from the mnist dataset
    s1 = GenerateCVFoldsStage(strategy='manual_train_test', strategy_args={'train_idx': range(mnist_train_df.shape[0]), 'test_idx': range(mnist_train_df.shape[0], mnist_df.shape[0])})
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
    tfm.set_fit_params({'epochs': 1, 'batch_size': 32})
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    tfm.set_compile_params({'optimizer': sgd, 'loss': 'CategoricalCrossentropy', 'metrics': 'accuracy'})
    tfm.set_model_params({'hidden_layer_size': 100})
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

    p.getDC().save('tf_test_results')

if __name__ == '__main__':
    RunML()
