"""Script of my solution to DrivenData Modeling Women's Health Care Decisions
Use this script in the following way:
    python solution.py <name-of-submission>
Argument is optional, the script will assign default name.
"""

from __future__ import division
import sys
import pdb

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import multiclass

from XGBoostClassifier import XGBoostClassifier


np.random.seed(17411)


def multiclass_log_loss(y_true, y_prob, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples, n_classes]
    y_prob : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_prob, eps, 1 - eps)
    rows = y_prob.shape[0]
    cols = y_prob.shape[1]
    vsota = np.sum(y_true * np.log(predictions) + (1-y_true) * np.log(1-predictions))
    vsota = vsota / cols
    return -1.0 / rows * vsota


def load_train_data(path=None, train_size=0.8):
    train_values = pd.read_csv('data/processed_train.csv')
    train_labels = pd.read_csv('data/train_labels.csv')

    df = pd.concat([train_values, train_labels], axis=1)
    df = df.drop('id', axis=1)
    X = df.values.copy()

    np.random.shuffle(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X[:, :-14], X[:, -14:], train_size=train_size,
    )

    print(" -- Data loaded.")
    return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(int), y_valid.astype(int))


def load_test_data(path=None):
    df = pd.read_csv('data/processed_train.csv')
    X = df.values
    X_test, ids = X[:, 1:], X[:, 0]
    return X_test.astype(float), ids.astype(str)


def validate(clf, model, X_train, X_valid, y_train, y_valid):
    """My local validation.
    My current best score is:
        - `0.2529` in Local Validation.
        - `0.2547` in Leaderboard.
    """
    print(" --- Evaluating {}.".format(model))
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_valid)
    score = multiclass_log_loss(y_valid, y_prob)
    print(" --- Multiclass logloss on validation set: {:.4f}".format(score))


def train():
    X_train, X_valid, y_train, y_valid = load_train_data()

    """About Xgboost Parameters.
    Because the distribution of each labels is not uniform. Each classifier may
    have outstanding accuracy that lead to overfit. So, increasing gamma to
    penalize that classifier to not overfit that label.

    More information about xgboost parameters: 
    https://github.com/dmlc/xgboost/wiki/Parameters

    So far, this parameters give score `0.2529` on local validation. And got
    `0.2547` at LB score. Using experimentation datasets.
    params =  
      - 'max_depth': 6
      - 'num_round': 512
      - 'gamma': 1.0
      - 'min_child_weight': 4
      - 'eta': 0.025
      - 'objective': 'binary:logistic'
      - 'eval_metric': 'logloss'
      - 'nthread': 4
    """
    model = "xgboost gbt"
    params = {'max_depth': 6,
              'num_round': 512,
              'gamma': 1.0,
              'min_child_weight': 4,
              'eta': 0.025,
              'objective': 'binary:logistic',
              'eval_metric': 'logloss',
              'nthread': 4}
    clf = XGBoostClassifier(**params)

    # Multilabel
    clf = multiclass.OneVsRestClassifier(clf, n_jobs=1)

    # Local Validation
    validate(clf, model, X_train, X_valid, y_train, y_valid)

    # Train whole set for submission.
    X = np.concatenate((X_train, X_valid))
    y = np.concatenate((y_train, y_valid))

    print(" --- Start training {} Classifier on whole set.".format(model))
    clf.fit(X, y)
    print(" --- Finished training on whole set.")

    print(" -- Finished training.")
    return clf


def make_submission(clf, path='my_submission.csv'):
    path = sys.argv[1] if len(sys.argv) > 1 else path
    X_test, ids = load_test_data()

    sample = pd.read_csv('data/SubmissionFormat.csv')

    y_prob = clf.predict_proba(X_test)
    preds = pd.DataFrame(y_prob, index=sample.id.values, columns=sample.columns[1:])

    preds.to_csv(path, index_label='id')
    print(" -- Wrote submission to file {}.".format(path))


def main():
    print(" - Start.")
    clf = train()
    make_submission(clf)
    print(" - Finished.")


if __name__ == '__main__':
    main()
