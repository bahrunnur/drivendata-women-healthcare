import sys

import numpy as np
import pandas as pd

"""
:: Preprocessing ::
-------------------
Fillin missing value. Dropping column that has no value.


: Release Feature :
-------------------
Binarizing Release Features, treat it like categorical features but it doesn't
contain any missing value in it.


: Numerical Features :
----------------------
Numerical features is column that has prefix: `n_XXXX`

First, find standard deviation for the respected column. If the `std_dev` value is
below 1 (Gaussian), impute the missing cell with mean value. But if not, impute
that with median value. If The missing value is more than 90%, fill that with -1.


: Ordinal Features :
--------------------
Ordinal features is column that has prefix: `o_XXXX`

If there is a column has NaN value, impute that with mode. And then treat it like
categorical features. Otherwise, just use the integer form, because there will
be so many features created from that.

Because there will be so many features. Another approach is to transform ordinal
into numeric via sigmoid function. But first, substract every value with median
(eg: 1-4, 2-4) to get mean value near zero.


: Categorical Features :
------------------------
Categorical features is column that has prefix: `c_XXXX`

Impute missing value with category `missing`. And then, binarizing it.

"""

# TODO: Feature engineering. Not selection. Do this first.
"""
:: Feature Engineering ::
-------------------------
Transform some features to create new features to improve model performance.
A way to overfit, but later I will do feature selection to underfit it.

Some approaches:

- Combine sparse (binary format/design matrix) categorical features to create
  hierarchy variable, this concept is similar to word n-gram feature engineering
  in text data.

        ex: (c_0879_a, c_0967_b), (c_0679_c, c_0680_d), ...

- Multiplicative Interactions in numerical features.

        ex: (n_0003 * n_0004), (n_0004 * n_0005), ...

- Less Common Variant in numerical feature. The challenge is how to deal with
  0 value in divider.

        ex: (n_0003 / n_0004), (n_0004 / n_0005), ...

- Binning numerical features and ordinal features. Decision Tree based classifier
  will do it automatically. But it worth to try to give more information into model.

- Engineer ordinal features? idk.

"""

# TODO: Feature selection from engineered features.
"""
:: Feature Selection ::
-----------------------
Select the best features from engineered features that have more impact at
model performance.

Approaches:

- Log model performance (using cross validation) based on combination of
  features selected. Dump performance data into file and analyze it.
  After that, make some selection.

- Forward Selection. Running algorithm on every features individually. Take
  the best, and keep that feature. Try to combine this kept feature with each other
  features. Take the best, keep it, repeat until no features can be added that
  makes the model better.

- Fast Correlation. Find the features that has high correlation with predicted
  value. After you got that feature, drop other features that has high correlation
  with that feature.

"""

def get_min_filled_threshold(df):
    """get minimum filled value count that allowed."""
    percentage = 0.1
    return df.shape[0] * percentage


def get_non_missing_count(col):
    return np.count_nonzero(~np.isnan(col.values))


def convert_categorical(df):
    """convert categorical column into binary features"""
    print(" --- Converting Categories into binary features.")
    columns = df.columns
    categorical = [x for x in columns if x.startswith('c_')]
    for col in categorical:
        print(" ---- Converting: {}".format(col))
        category_binary = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, category_binary], axis=1)
    df = df.drop(categorical, axis=1)
    print(" --- Finished converting Categories into binary features.")
    return df


def fill_nan_in_category(df):
    """fill NaN with mode"""
    print(" --- Filling NaN in Categories.")
    columns = df.columns
    categorical = [x for x in columns if x.startswith('c_')]
    df[categorical] = df[categorical].fillna('missing')
    print(" --- Finished filling NaN in Categories.")
    return df


def fill_nan_in_numeric(df):
    """fill NaN value with mean or median or -1 (below thresh)"""
    print(" --- Filling NaN in Numerics.")
    thresh = get_min_filled_threshold(df)
    columns = df.columns
    numerical = [x for x in columns if x.startswith('n_')]
    # fill NaN with mean or median, based on std dev
    for col in numerical:
        filled = get_non_missing_count(df[col])
        if filled < thresh:
            df[col] = df[col].fillna(-1)
        else:
            std = df[col].std()
            if std < 1:
                mean = df[col].mean()
                df[col] = df[col].fillna(mean)
            else:
                median = df[col].median()
                df[col] = df[col].fillna(mean)

    print(" --- Finished filling NaN in Numerics.")
    return df


def sigmoid(x):
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def deal_with_ordinal(df):
    """Dealing with ordinal features column,
    fill missing value with mode.
    """
    print(" --- Dealing with Ordinals.")
    thresh = get_min_filled_threshold(df)
    columns = df.columns
    ordinal = [x for x in columns if x.startswith('o_')]

    for col in ordinal:
        filled = get_non_missing_count(df[col])
        if filled < thresh:
            df[col] = df[col].fillna(-10)
        else:
            mode = df[col].value_counts().idxmax()
            df[col] = df[col].fillna(mode)
        median = df[col].median()
        df[col] = df[col].apply(lambda x: sigmoid(x-median))
    
    print(" --- Finished dealing with Ordinals.")
    return df


def process():
    print(" -- Processing Data.")
    trains = pd.read_csv('data/train_values.csv', low_memory=False)
    tests = pd.read_csv('data/test_values.csv', low_memory=False)
    # split based on df index
    split = trains.index.values.max() + 1

    df = pd.concat([trains, tests], axis=0)

    # :: DEALING WITH MISSING VALUE ::
    # =========================================================================
    # drop nothing.
    df = deal_with_ordinal(df)
    df = fill_nan_in_numeric(df)
    df = fill_nan_in_category(df)
    # =========================================================================

    # :: GENERAL ::
    # =========================================================================
    # binarize categorical features
    df = convert_categorical(df)

    # binarize release feature
    print(" --- Converting release into binary features.")
    release_binary = pd.get_dummies(df.release, prefix=df.release.name)
    df = df.drop('release', axis=1)
    df = pd.concat([df, release_binary], axis=1)
    print(" --- Finished converting release into binary features.")
    # =========================================================================

    # Split df, based on trains id and tests id
    proc_trains = df.iloc[:split]
    proc_tests = df.iloc[split:]

    # Persist it to file
    proc_trains.to_csv('data/processed_train.csv', index=False)
    proc_tests.to_csv('data/processed_test.csv', index=False)
    print(" -- Finished Processing Data.")


def main():
    print(" - Start.")
    process()
    print(" - Finished.")


if __name__ == "__main__":
    main()
