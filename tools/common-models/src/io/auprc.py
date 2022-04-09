from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

varname = "IsDisciplinaryStatement"

df = pd.read_csv("/home/cat/cetd-results/rf-compare-feature-sets2/predictions/RandomForestClassifier/"
                 "{}-AVERAGED-predictions.csv".format(varname)
                 )

#varname += " Hierarchical"
df["True_value.x"] = np.where(df["True_value.x"]  >= 'disciplinary', 1, 0)

#y_test, y_scores = df["True_value"].values, df["authentic question_probability"].values

y_test, y_scores = df["True_value.x"].values, df["avg_two"].values
average_precision = average_precision_score(y_test, y_scores)
auc_score = roc_auc_score(y_test, y_scores)

print('Average precision-recall score: {0:0.4f}'.format(
      average_precision))
print('AUC: {0:0.4f}'.format(
      auc_score))

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(varname + ': AUPRC={0:0.4f}'.format( average_precision))

plt.show()


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]


def precision_recall_threshold(p, r, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """

    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'],
                       index=['neg', 'pos']))

    # plot the curve
    #plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
             markersize=15)

    plt.show()


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(8, 8))
    plt.title("AuthCogUptake: Prec/recall by decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')

    plt.show()

plot_precision_recall_vs_threshold(precision, recall, thresholds )
#precision_recall_threshold(precision, recall, thresholds, .2)