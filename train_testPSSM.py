import os
import numpy as np
from sklearn import svm
#from sklearn.model_selection import train_test_split
import pickle

from PSSM_parser import parse_fasta, PSSM_parser, PSSM_X_vector, train_inputvector_X, inputvector_y

#X = train_inputvector_X(parse_fasta('../project/datasets/data.txt'))
#y = inputvector_y(parse_fasta('../project/datasets/data.txt'))

features, topology = train_inputvector_X('datasets/PSSM_files/test_PSSM')
X = features
y = inputvector_y('datasets/PSSM_files/test_PSSM')


print(X.shape)
#create model and give other than dtype:
model = svm.SVC(decision_function_shape='ovo')
#training data:
model.fit(X, y)
accuracy = model.score(X, y)
print(accuracy)

#using pickle to save model to disk:
filename = '../project/models/PSSM_win15_model.sav'
pickle.dump(model, open(filename, 'wb'))




#if __name__ == '__main__':
    #result_train_X = train_inputvector_X(parse_fasta('../project/datasets/data.txt'))
    #print (result_train_X)

    #result_y = inputvector_y(parse_fasta('../project/datasets/data.txt'))
    #print (result_y)
