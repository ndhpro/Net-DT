from __future__ import print_function
import argparse
import logging
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, mutual_info_classif

from report import report, draw_roc


def prepare_data(seed):
    with open('data/kddcup.names', 'r') as f:
        features = f.read().split('\n')
    features.append('label')
    kdd = pd.read_csv('data/kddcup.data_10_percent_corrected', names=features)
    kdd.loc[kdd['label'] == 'normal.', 'label'] = 1
    kdd.loc[kdd['label'] != 1, 'label'] = 0
    enc = LabelEncoder()
    kdd['protocol_type'] = enc.fit_transform(kdd['protocol_type'])
    kdd['service'] = enc.fit_transform(kdd['service'])
    kdd['flag'] = enc.fit_transform(kdd['flag'])
    X = kdd.values[:, :-1]
    y = kdd.values[:, -1]

    print(kdd.head())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed, test_size=0.3, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description='Machine Learning')
    parser.add_argument('-s', '--seed', type=int, default=2020,
                        metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--test', action='store_true',
                        default=False, help='training or testing mode')
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S.')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    y_true = dict()
    y_pred = dict()
    names = ["DT_Entropy", "DT_Gini", "RF_Entropy", "RF_Gini"]
    classifiers = [
        DecisionTreeClassifier(random_state=args.seed, criterion='entropy'),
        DecisionTreeClassifier(random_state=args.seed, criterion='gini'),
        RandomForestClassifier(random_state=args.seed, n_estimators=10,
                               n_jobs=-1, criterion='entropy'),
        RandomForestClassifier(random_state=args.seed, n_estimators=10,
                               n_jobs=-1, criterion='gini'),
    ]
    hyperparam = [
        {'splitter': ['best', 'random'],
            'max_features': ['sqrt', 'log2', None]},
        {'splitter': ['best', 'random'],
            'max_features': ['sqrt', 'log2', None]},
        {},
        {},
    ]
    colors = ['blue', 'orange', 'green', 'red',
              'purple', 'brown', 'pink', 'gray']

    X_train, y_train, X_test, y_test = prepare_data(args.seed)
    logger.info(str(X_train.shape) + ' ' + str(X_test.shape))

    for name, est, hyper in zip(names, classifiers, hyperparam):
        logger.info(name + '...')
        if not args.test:
            clf = GridSearchCV(est, hyper, cv=5, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_true[name],  y_pred[name] = y_test, clf.predict(X_test)
            logger.info('____Accuracy: %0.4f' %
                        metrics.accuracy_score(y_true[name], y_pred[name]))
            print(clf.best_estimator_)
            pickle.dump(clf, open('model/kdd99_' + str(name) + '.sav', 'wb'))
        else:
            clf = pickle.load(open('model/kdd99_' + str(name) + '.sav', 'rb'))
            y_true[name], y_pred[name] = y_test, clf.predict(X_test)

    report('kdd99', names, y_true, y_pred)
    draw_roc('kdd99', names, colors, y_true, y_pred)


if __name__ == "__main__":
    main()
