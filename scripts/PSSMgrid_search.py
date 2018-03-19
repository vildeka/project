import numpy as np
import pandas as pd
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

from PSSM_parser import parse_fasta, PSSM_parser, PSSM_X_vector, train_inputvector_X, inputvector_y

def optimize_params(filename_train, directory_train, filename_test, directory_test):
    features, topology = train_inputvector_X(filename_train, directory_train)
    X_train = features
    y_train = inputvector_y(filename_train, directory_train)

    features, topology = train_inputvector_X(filename_train, directory_train)
    X_test = features
    y_test = inputvector_y(filename_train, directory_train)

    #svc = LinearSVC()
    #parameters = {'C':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
    #svc = svm.SVC(C=2.3,decision_function_shape='ovr', gamma=0.05)
    #parameters = {'C':[ 2, 2.5], 'gamma':[0.01, 0.05,]}
    svc = RandomForestClassifier()
    parameters = {'n_estimators':[200, 300, 400], 'min_samples_split':[2, 3, 4], 'class_weight':['balanced']}
    model = GridSearchCV(svc, parameters, cv=3, verbose=True, refit=False, return_train_score=False, n_jobs=-1)
    model.fit(X_train, y_train)
    res = pd.DataFrame(model.cv_results_)

    res.to_csv('../project/results/grid_serchforestPSSM.csv', sep='\t', encoding='UTF-8')
    print('Grid serch done')

if __name__ == '__main__':
    optimize_params('../project/datasets/train_data0.7.txt', '../project/datasets/PSSM_files/train_0.7',
                    '../project/datasets/test_data0.3.txt', '../project/datasets/PSSM_files/test_0.3')
