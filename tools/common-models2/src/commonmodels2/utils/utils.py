from sklearn.metrics import get_scorer, SCORERS
from tensorflow.keras import losses
from tensorflow.keras.losses import get as get_loss
from tensorflow.keras.metrics import get as get_metric
from tensorflow.keras.optimizers import get as get_optimizer
import torch

def get_any_scoring_func(func_str):
    try:
        scoring_func = get_sklearn_scoring_func(func_str)
    except:
        try:
            scoring_func =  get_tensorflow_loss_func(func_str)
        except:
            raise RuntimeError('Unable to map string "{}" to scoring function'.format(func_str))
    return scoring_func

def get_sklearn_scoring_func(scoring_str):
    if scoring_str in SCORERS.keys():   
        scorer = get_scorer(scoring_str)
        # BB - Accessing the private _score_func can lead to weird side-effects! Please be aware
        # this approach may not work for all scoring functions:
        # https://stackoverflow.com/questions/63943410/getting-a-scoring-function-by-name-in-scikit-learn
        scoring_ref = scorer._score_func
        if scoring_str.startswith('neg_'):
            scoring_ref = lambda x,y: -scoring_ref(x,y)
        return scoring_ref
    else:
        raise ValueError("Scoring func {} not supported by ScikitLearn".format(scoring_str))

# BB: Tensorflow's get() function for losses returns two kinds of objects.  If the class name string is passed in
#     (e.g., "MeanAbsoluteError"), then you get a callable class object that evaluates to a scalar.  If the string
#     is the function name (e.g., "mean_absolute_error"), then you get a callable function that computes the
#     sample-wise error and returns a vector.  This function returns the callable class version.
def get_tensorflow_loss_func(loss_str, config=None):
    tensorflow_loss_strs = [
        'BinaryCrossentropy',
        'CategoricalCrossentropy',
        'CategoricalHinge',
        'CosineSimilarity',
        'Hinge',
        'Huber',
        'KLDivergence',
        'LogCosh',
        'MeanAbsoluteError',
        'MeanAbsolutePercentageError',
        'MeanSquaredError',
        'MeanSquaredLogarithmicError',
        'Poisson',
        'SparseCategoricalCrossentropy',
        'SquaredHinge'
         ]

    if loss_str in tensorflow_loss_strs:
        if config is not None:
            loss_class = type(get_loss(loss_str)).__name__
            get_loss_dict = {'class_name': loss_class, 'config': config}
            loss_fn = get_loss(get_loss_dict)
        else:
            loss_fn = get_loss(loss_str)
        return loss_fn
    else:
        raise ValueError('Loss function "{}" not supported by Tensorflow'.format(loss_str))
        
    
# BB: This metric getter returns the callable class version instead of the function.  See the comment for
#     get_tensorflow_loss_func() above.
def get_tensorflow_metric_func(metric_str, config=None):
    tensorflow_metric_strs = [
        'AUC',
        'Accuracy',
        'BinaryAccuracy',
        'BinaryCrossentropy',
        'CategoricalAccuracy',
        'CategoricalCrossentropy',
        'CategoricalHinge',
        'CosineSimilarity',
        'FalseNegatives',
        'FalsePositives',
        'Hinge',
        'KLDivergence',
        'LogCoshError',
        'Mean',
        'MeanAbsoluteError',
        'MeanAbsolutePercentageError',
        'MeanIoU',
        'MeanRelativeError',
        'MeanSquaredError',
        'MeanSquaredLogarithmicError',
        'MeanTensor',
        'Poisson',
        'Precision',
        'PrecisionAtRecall',
        'Recall',
        'RecallAtPrecision',
        'RootMeanSquaredError',
        'SensitivityAtSpecificity',
        'SparseCategoricalAccuracy',
        'SparseCategoricalCrossentropy',
        'SparseTopKCategoricalAccuracy',
        'SpecificityAtSensitivity',
        'SquaredHinge',
        'Sum',
        'TopKCategoricalAccuracy'
        'TrueNegatives',
        'TruePositives'
         ]

    if metric_str in tensorflow_metric_strs:
        if config is not None:
            metric_class = type(get_metric(metric_str)).__name__
            get_metric_dict = {'class_name': metric_class, 'config': config}
            metric_fn = get_metric(get_metric_dict)
        else:
            metric_fn = get_metric(metric_str)
        return metric_fn
    else:
        raise ValueError('Metric function "{}" not recognized and may not be supported by Tensorflow'.format(metric_str))

# BB: This optimizer getter returns the callable class version instead of the function.  See the comment for
#     get_tensorflow_loss_func() above.
def get_tensorflow_optimizer_func(optimizer_str):
    tensorflow_optimizer_strs = [
        'adadelta',
        'adagrad',
        'adam',
        'adamax',
        'ftrl',
        'nadam',
        'rmsprop',
        'sgd'
    ]

    if optimizer_str.lower() in tensorflow_optimizer_strs:
        return get_optimizer(optimizer_str.lower())
    else:
        raise ValueError('Optimizer function "{}" not recognized and may not be supported by Tensorflow'.format(optimizer_str))

def get_torch_loss_func(loss):
    torch_loss_strs = {
        'binary_crossentropy': torch.nn.BCELoss,
        'binary_crossentropy_with_logits': torch.nn.BCEWithLogitsLoss,
        'mean_absolute_error': torch.nn.L1Loss,
        'mean_squared_error': torch.nn.MSELoss,
        'categorical_crossentropy': torch.nn.CrossEntropyLoss,
        'kl_divergence': torch.nn.KLDivLoss,
        'kld': torch.nn.KLDivLoss,
        'kullback_leibler_divergence': torch.nn.KLDivLoss,
        'smooth_l1_loss': torch.nn.SmoothL1Loss
    }
    try:
        if issubclass(loss,torch.nn.Module):
            return loss
    except TypeError:
        pass
    if isinstance(loss,str):
        if loss in torch_loss_strs:
            return torch_loss_strs[loss]
        else:
            raise ValueError(f"Couldn't map string to pytorch loss function. Try one of the following: {torch_loss_strs} or pass in a torch.nn loss function class")
    else:
        raise ValueError(f"Input isn't a torch.nn loss function or a string. Please pass in one or the other.")

def get_torch_optimizer(optimizer):
    torch_optimizer_strs = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adamw': torch.optim.AdamW,
        'adamax': torch.optim.Adamax,
        'lbfgs': torch.optim.LBFGS,
        'rmsprop': torch.optim.RMSprop
    }
    try:
        if issubclass(optimizer, torch.optim.Optimizer):
            return optimizer
    except TypeError:
        pass
    if isinstance(optimizer,str):
        if optimizer in torch_optimizer_strs:
            return torch_optimizer_strs[optimizer]
        else:
            raise ValueError(f"Couldn't map string to pytorch optimizer. Try one of the following: {torch_optimizer_strs} or pass in a torch.optim.Optimizer class")
    else:
        raise ValueError(f"Optimizer must be a torch.optim.Optimizer subclass or a string")
        
