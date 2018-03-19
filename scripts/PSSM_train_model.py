import os
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
import pickle

from PSSM_parser import parse_fasta, PSSM_parser, PSSM_X_vector, train_inputvector_X, inputvector_y

def train_model(filename, directory):
    features, topology = train_inputvector_X(filename, directory)

    X = features
    y = inputvector_y(filename, directory)

    shape = print(X.shape)
    #create model and give other than dtype:
    model = svm.SVC(C=2.3,decision_function_shape='ovr', gamma=0.05)
    #model = LinearSVC()
    #model = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

    #training data:
    model.fit(X, y)
    accuracy = model.score(X, y)
    ac_score = print(accuracy)

    #using pickle to save model to disk:
    filename = '../project/models/PSSM_SVM_y_win15_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return shape, ac_score



if __name__ == '__main__':
    train_model('../project/datasets/train_data0.7.txt', '../project/datasets/PSSM_files/train_0.7')
    #train_model('datasets/PSSM_files/test_PSSM/PSSM_test.fasta', 'datasets/PSSM_files/test_PSSM')
