from __future__ import print_function
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


def draw_roc(out, names, colors, y_true, y_pred):
    plt.figure()
    for name, color in zip(names, colors):
        fpr, tpr, _ = metrics.roc_curve(y_true[name], y_pred[name])
        auc = metrics.roc_auc_score(y_true[name], y_pred[name])
        plt.plot(fpr, tpr, color=color,
                 label='%s ROC (area = %0.3f)' % (name, auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('log/' + out + '_roc.png')


def report(out, names, y_true, y_pred):
    with open('log/' + out + '_result.txt', 'w') as f:
        for name in names:
            clf_report = metrics.classification_report(
                y_true[name], y_pred[name], digits=4)
            cnf_matrix = metrics.confusion_matrix(y_true[name], y_pred[name])
            TN, FP, FN, TP = cnf_matrix.ravel()
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            f.write(str(name) + ':\n')
            f.write('Accuracy: %0.4f\n' %
                    metrics.accuracy_score(y_true[name], y_pred[name]))
            f.write('ROC AUC: %0.4f\n' %
                    metrics.roc_auc_score(y_true[name], y_pred[name]))
            f.write('TPR: %0.4f\nFPR: %0.4f\n' % (TPR, FPR))
            f.write('Classification report:\n' + str(clf_report) + '\n')
            f.write('Confusion matrix:\n' + str(cnf_matrix) + '\n')
            draw_confusion_matrix(out, name, cnf_matrix)
            f.write(64*'-'+'\n')


def draw_confusion_matrix(out, title, cm):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    plt.title('Confusion matrix (' + title + ')')
    plt.colorbar()
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig('log/' + out + '_conf_matrix_' + title)
