# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 16:45:33 2018

@author: bonfardeci-j
"""

import pandas as pd
import numpy as np
from itertools import cycle
from sklearn.metrics import roc_curve, auc, confusion_matrix
import re
import statsmodels.api as sm
import os
from copy import copy
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.feature_selection import RFE, SelectFromModel
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Utilities:

    @staticmethod
    def get_balanced_accuracy(tpr, fpr):
        """
        Get Balanced Accuracy - an average of the true positive rate and false positive rate.
        Dr. Alan Dabney

        @param tpr: float (True Positive Rate - the Sensitivity)
        @param fpr: float (False Positive Rate - the 1-Specificity)
        @returns float
        """
        return (tpr + (1-fpr)) / 2

    @staticmethod
    def get_confusion_matrix(cutoff, actual, prob):
        """
        Return a confusion matrix with the optimal threshold/cutoff for probability of Y.

        TN | FP
        -------
        FN | TP

        For example: 

           n=165   | Predicted NO | Predicted YES
        ------------------------------------------
        Actual NO  |       50     |      10       |
        ------------------------------------------
        Actual YES |       5      |      100      |
        ------------------------------------------

        The diagonal elements represent the number of points for which the predicted label is equal to the true label,
        while off-diagonal elements are those that are mislabeled by the classifier.
        The higher the diagonal values of the confusion matrix the better, indicating many correct predictions.
        @param cutoff <float>
        @param actual <list>
        @param prob <list>
        @returns 2D array <list<list>>
        """
        pred = []
        for (x, y) in prob:
            pred.append(1 if y >= cutoff else 0)

        return confusion_matrix(actual, pred)

    @staticmethod
    def get_tpr_fpr(cm):
        """
        Sensitivity: TruePos / (True Pos + False Neg) 
        Specificity: True Neg / (False Pos + True Neg)
        TN | FP
        -------
        FN | TP
        @param 2D array <list<list>>
        @returns <list<float>>
        """

        tn = float(cm[0][0])
        fp = float(cm[0][1])
        fn = float(cm[1][0])
        tp = float(cm[1][1])

        tpr = tp / (tp + fn)
        fpr = 1-(tn / (fp + tn))

        return [tpr, fpr] 

    @staticmethod
    def get_best_cutoff(actual, prob):  
        """
        Get the best cutoff according to Balanced Accuracy
        'Brute-force' technique - try all cutoffs from 0.01 to 0.99 in increments of 0.01

        @param actual <list<float>>
        @param prob <list<tuple<float, float>>>
        @returns <list<float>>
        """
        best_tpr = 0.0; best_fpr = 0.0; best_cutoff = 0.0; best_ba = 0.0; 
        cutoff = 0.0
        cm = [[0,0],[0,0]]
        while cutoff < 1.0:
            _cm = Utilities.get_confusion_matrix(cutoff=cutoff, actual=actual, prob=prob)
            _tpr, _fpr = Utilities.get_tpr_fpr(_cm)

            if(_tpr < 1.0):    
                ba = Utilities.get_balanced_accuracy(tpr=_tpr, fpr=_fpr)

                if(ba > best_ba):
                    best_ba = ba
                    best_cutoff = cutoff
                    best_tpr = _tpr
                    best_fpr = _fpr
                    cm = _cm

            cutoff += 0.01

        tn = cm[0][0]; fp = cm[0][1]; fn = cm[1][0]; tp = cm[1][1];
        return [best_tpr, best_fpr, best_cutoff, tn, fp, fn, tp]

    @staticmethod
    def show_confusion_matrix(C, class_labels=['0','1'], figsize=(6,6), fontsize=12, filename='roc-curve'):
        """
        C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
        class_labels: list of strings, default simply labels 0 and 1.
        Sensitivity: TruePos / (True Pos + False Neg) 
        Specificity: True Neg / (False Pos + True Neg)
        TN | FP
        -------
        FN | TP
        Draws confusion matrix with associated metrics.
        https://notmatthancock.github.io/2015/10/28/confusion-matrix.html
        """
        assert C.shape == (2,2), "Confusion matrix should be from binary classification only."

        # true negative, false positive, etc...
        tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];
        sensitivity = tp / (tp + fn)
        specificity = tn / (fp + tn)
        precision = tp / (tp+fp)

        NP = fn+tp # Num positive examples
        NN = tn+fp # Num negative examples
        #N  = NP+NN

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111)
        ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

        # Draw the grid boxes
        ax.set_xlim(-0.5,2.5)
        ax.set_ylim(2.5,-0.5)
        ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
        ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
        ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
        ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

        # Set xlabels
        ax.set_xlabel('Predicted', fontsize=fontsize)
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(class_labels + [''])
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        # These coordinate might require some tinkering. Ditto for y, below.
        ax.xaxis.set_label_coords(0.34,1.06)

        # Set ylabels
        ax.set_ylabel('Actual', fontsize=fontsize, rotation=90)
        ax.set_yticklabels(class_labels + [''],rotation=90)
        ax.set_yticks([0,1,2])
        ax.yaxis.set_label_coords(-0.09,0.65)


        # Fill in initial metrics: tp, tn, etc...
        ax.text(0,0,
                'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
                va='center',
                ha='center',
                bbox=dict(fc='w',boxstyle='round,pad=1'))

        ax.text(0,1,
                'False Neg: %d'%fn,
                va='center',
                ha='center',
                bbox=dict(fc='w',boxstyle='round,pad=1'))

        ax.text(1,0,
                'False Pos: %d'%fp,
                va='center',
                ha='center',
                bbox=dict(fc='w',boxstyle='round,pad=1'))


        ax.text(1,1,
                'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
                va='center',
                ha='center',
                bbox=dict(fc='w',boxstyle='round,pad=1'))

        # Fill in secondary metrics: accuracy, true pos rate, etc...
        ax.text(2,0,
                'False Pos Rate: {:2.4%}\n(True Neg Rate. {:2.3%})'\
                .format(1-specificity, specificity),
                va='center',
                ha='center',
                bbox=dict(fc='w',boxstyle='round,pad=1'))

        ax.text(2,1,
                'True Pos Rate: {:2.4%}'.format(sensitivity),
                va='center',
                ha='center',
                bbox=dict(fc='w',boxstyle='round,pad=1'))

        ax.text(2,2,
                'Precision: %.4f'%(precision),
                va='center',
                ha='center',
                bbox=dict(fc='w',boxstyle='round,pad=1'))

        ax.text(0,2,
                'Neg Pre Val: %.4f'%(1-fn/(fn+tn+0.)),
                va='center',
                ha='center',
                bbox=dict(fc='w',boxstyle='round,pad=1'))

        ax.text(1,2,
                'Pos Pred Val: %.4f'%(tp/(tp+fp+0.)),
                va='center',
                ha='center',
                bbox=dict(fc='w',boxstyle='round,pad=1'))


        filename = '%s.png' % (filename)
        plt.savefig(filename)
        print('Confusion matrix image was saved to: %s' % (filename))

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def show_roc(kfolds, mean_tpr, mean_fpr, title, lw=2, filename='roc'):
        """
        Display and save ROC curve
        """
        plt.figure(figsize=(12,12))
        colors = cycle(['cyan', 'red', 'seagreen', 'darkorange', 'blue'])

        # Plot the ROC Curve for this CV group
        i=0
        for (k, color) in zip(kfolds, colors):
            tpr, fpr = k[0], k[1]
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))
            i += 1

        # Plot the ROC Curve for logistic regression
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

        mean_tpr /= len(kfolds)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive (1-Specificity)')
        plt.ylabel('True Positive (Sensitivity)')
        plt.title(title)
        plt.legend(loc="lower right")

        filename = '%s.png' % (filename)
        plt.savefig(filename)
        print('ROC image was saved to: %s' % (filename))

        plt.show()

    @staticmethod
    def output_cms(filename, data, sheetname='Sheet1'):
        """
        Output Cutoffs, TPR, FPR, Confusion, ... to Excel 
        """

        cols = ['Method', 'FoldNum', 'Cutoff', 'Sensitivity', '1-Specificity', 
                'TrueNeg', 'FalsPos', 'FalseNeg', 'TruePos', 'Accuracy', 'AUC', 'Balanced Accuracy']

        df = pd.DataFrame(data=data, columns=cols)
        df.sort_values(by=['FoldNum', 'Cutoff'])
        filename = '%s.csv'%(filename)
        df.to_csv(filename, sep=',', encoding='utf-8')
        print('Excel file was saved to: %s' % (filename))
     
    @staticmethod
    def combine(prob, x, y, predicted):
        data = []    
        i = 0
        for xrow, yrow, prob, pred in zip(x, y, prob, predicted):
            a = []
            for col in xrow:
                a.append(col)

            a.append(yrow)
            a.append(prob[0])
            a.append(prob[1])
            a.append(pred)
            data.append(a)
            i+=1

        return data
    
    # Lasso feature selection
    @staticmethod
    def get_lasso_selection(X, y, columns):
        """
        Select significant variables according to Lasso.
        Best model is selected by cross-validation.
        @param X <Pandas Dataframe>
        @param y <list>
        @columns <list>
        @returns <list>
        """
        clf = LassoCV(max_iter=10000)
        sfm = SelectFromModel(clf)
        sfm.fit(X, y)
        #features = sfm.transform(X).shape[1]
        feature_indices = sfm.get_support()
        significant_features = []
        for c, b in zip(columns, feature_indices):
            if b:
                significant_features.append(c)
                
        return significant_features
        
    @staticmethod
    def remove_insignificant_vars(sig_var_list, df):
        """
        Remove insignificant variables from a DataFrame.
        @param sig_var_list <list>
        @param df <DataFrame>
        @returns <DataFrame>
        """
        drop = []
        cols = df.columns.tolist()
        for v in cols:
            if not v in sig_var_list:
                drop.append(v)
                df = df.drop(str(v), 1)
                
           
        print('\r\nDropped insignificant vars: ', ', '.join(drop))
        return df
