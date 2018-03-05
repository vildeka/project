from sklearn import svm
#from sklearn.model_selection import train_test_split
import pickle

from dataparser import parse_fasta, train_inputvector_X, inputvector_y

#X = train_inputvector_X(parse_fasta('../project/data.txt'))
#y = inputvector_y(parse_fasta('../project/data.txt'), window=3)

X = train_inputvector_X(parse_fasta('../project/dataminix2.txt'))
y = inputvector_y(parse_fasta('../project/dataminix2.txt'), window=3)

print(X.shape)
#create model and give other than dtype: 
model = svm.SVC(decision_function_shape='ovo')
#training data:
model.fit(X, y)
accuracy = model.score(X, y)
print(accuracy)

#using pickle to save model to disk:
filename = "minix2_model.sav"
pickle.dump(model, open(filename, 'wb'))










#https://github.com/WangXueqing007/KB8024-Bioinformatics-Project
#https://github.com/carolinasavatier


#http://scikit-learn.org/stable/modules/svm.html
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
#https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

