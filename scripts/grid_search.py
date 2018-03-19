import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve

from dataparser import parse_fasta, inputvector_X, inputvector_y, train_inputvector_X

def optimize_params(x_train, y_train):
    x_test = train_inputvector_X(parse_fasta('../project/datasets/test_data0.3.txt'))
    y_test = inputvector_y(parse_fasta('../project/datasets/test_data0.3.txt'))

    #svc = LinearSVC()
    #parameters = {'C':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
    #svc = svm.SVC(C=2.3,decision_function_shape='ovr', gamma=0.05, class_weight='balanced')
    #parameters = {'C':[ 2, 2.5], 'gamma':[0.01, 0.05,]}
    svc = RandomForestClassifier()
    #parameters = {'n_estimators':[ 50, 75, 100], 'min_samples_split':[2, 3, 4], 'class_weight':['balanced']}
    parameters = {'n_estimators':[300], 'min_samples_split':[3]}
    model = GridSearchCV(svc, parameters, cv=3, verbose=True, refit=False, return_train_score=False, n_jobs=-1)
    model.fit(x_train, y_train)
    res = pd.DataFrame(model.cv_results_)

    res.to_csv('../project/results/grid_serchrandom.csv', sep='\t', encoding='UTF-8')
    print('Grid serch done')


if __name__ == '__main__':
    optimize_params(train_inputvector_X(parse_fasta('../project/datasets/train_data0.7.txt')),
                        inputvector_y(parse_fasta('../project/datasets/train_data0.7.txt')))
