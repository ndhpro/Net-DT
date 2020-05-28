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
    nb_train = pd.read_csv('data/UNSW_NB15_training-set.csv')
    nb_test = pd.read_csv('data/UNSW_NB15_testing-set.csv')

    nb_train = nb_train.drop(columns=['id', 'attack_cat'])
    nb_test = nb_test.drop(columns=['id', 'attack_cat'])

    enc = LabelEncoder()
    nb_train['proto'] = enc.fit_transform(nb_train['proto'])
    nb_train['service'] = enc.fit_transform(nb_train['service'])
    nb_train['state'] = enc.fit_transform(nb_train['state'])

    nb_test['proto'] = enc.fit_transform(nb_test['proto'])
    nb_test['service'] = enc.fit_transform(nb_test['service'])
    nb_test['state'] = enc.fit_transform(nb_test['state'])

    X_train, y_train = nb_train.values[:, :-1], nb_train.values[:, -1]
    X_test, y_test = nb_test.values[:, :-1], nb_test.values[:, -1]

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
            pickle.dump(clf, open('model/unsw-nb15_' + str(name) + '.sav', 'wb'))
        else:
            clf = pickle.load(open('model/unsw-nb15_' + str(name) + '.sav', 'rb'))
            y_true[name], y_pred[name] = y_test, clf.predict(X_test)

    report('unsw-nb15', names, y_true, y_pred)
    draw_roc('unsw-nb15', names, colors, y_true, y_pred)


if __name__ == "__main__":
    main()
