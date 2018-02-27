import pandas as pd
from sklearn import svm
#from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle

from dataparser import input_vectors, parse_fasta

#X, y = input_vectors(parse_fasta('../project/data.txt'))
X, y = input_vectors(parse_fasta("datamini.txt"))

print(X.shape)
#create model and give other than dtype: 
model = svm.SVC(decision_function_shape='ovo')
#training data:
model.fit(X, y)
accuracy = model.score(X, y)
print(accuracy)

#using pickle to save model to disk:
filename = "first_model.sav"
pickle.dump(model, open(filename, 'wb'))










#https://github.com/WangXueqing007/KB8024-Bioinformatics-Project
#https://github.com/carolinasavatier


#http://scikit-learn.org/stable/modules/svm.html
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
#https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

