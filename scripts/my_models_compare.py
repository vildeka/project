import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

from PSSM_parser import parse_fasta, PSSM_parser, PSSM_X_vector, train_inputvector_X, inputvector_y

print('loading X and y vector...')
#creates the X and y input vectors from the dataset:
features, topology = train_inputvector_X('../project/datasets/train_data0.7.txt', '../project/datasets/PSSM_files/train_0.7')
X = features
y = inputvector_y('../project/datasets/train_data0.7.txt', '../project/datasets/PSSM_files/train_0.7')

features, topology = train_inputvector_X('../project/datasets/train_data0.7.txt', '../project/datasets/PSSM_files/train_0.7')
X_train = features
y_train = inputvector_y('../project/datasets/train_data0.7.txt', '../project/datasets/PSSM_files/train_0.7')

features, topology = train_inputvector_X('../project/datasets/test_data0.3.txt', '../project/datasets/PSSM_files/test_0.3')
X_test = features
y_test = inputvector_y('../project/datasets/test_data0.3.txt', '../project/datasets/PSSM_files/test_0.3')

#uses sklearn module to generate train and test sets by splitting original dataset:
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


#Training and testing the different classifiers SVM, decisiontree and randomforest
labels = [1, 2, 3]
states = ['H', 'S', 'C']

#SVM classifier
model = pickle.load(open('../project/models/PSSM_SVM_y_win15_Balanced_model.sav', 'rb'))
svm_cross_score = cross_val_score(model, X, y, cv=10, verbose=True, n_jobs=-1)
print('SVM cross validation done...')
svm_cross_mean = svm_cross_score.mean()
#model.fit(X_train, y_train)
print('SVM training done...')

svm_y_predicted = model.predict(X_test)
print(svm_y_predicted)
svm_classreport = classification_report(y_test, svm_y_predicted, labels=labels, target_names=states)
svm_confusionm = confusion_matrix(y_test, svm_y_predicted, labels = labels)
svm_mcc = matthews_corrcoef(y_test, svm_y_predicted)

#decision tree classifier
model = DecisionTreeClassifier()
tree_cross_score = cross_val_score(model, X, y, cv=5, verbose=True, n_jobs=-1)
print('Decision tree cross validation done...')
tree_score_mean =  tree_cross_score.mean()
model.fit(X_train, y_train)
print('Decision tree traning done...')

tree_y_predicted = model.predict(X_test)
tree_classreport = classification_report(y_test, tree_y_predicted, labels=labels, target_names=states)
tree_confusionm = confusion_matrix(y_test, tree_y_predicted, labels=labels)
tree_mcc = matthews_corrcoef(y_test, tree_y_predicted)

#random forest classifier
model = pickle.load(open('../project/models/PSSM_random_win15_Balanced_model.sav', 'rb'))
random_cross_score = cross_val_score(model, X, y, cv = 5, verbose=True, n_jobs=-1)
print('Random forest cross validation done...')
random_score_mean = random_cross_score.mean()
#model.fit(X_train, y_train)
print('Random forest training finished')

random_y_predicted = model.predict(X_test)
random_classreport = classification_report(y_test, random_y_predicted, labels=labels, target_names=states)
random_confusionm = confusion_matrix(y_test, random_y_predicted, labels=labels)
random_mcc = matthews_corrcoef(y_test, random_y_predicted)

#prints results to screen:
print('Cross-validation scores for SVC: ' + str(svm_cross_mean)+ '\n')
print('Cross-validation scores for DecisionTreeClassifier: '+ str(tree_score_mean)+ '\n')
print('Cross-validation scores for RandomForestClassifier: '+ str(random_score_mean)+ '\n')
print('Matthews correlation coefficient (MCC) SVM: ' + str(svm_mcc) + '\n')
print('Matthews correlation coefficient (MCC) DecisionTreeClassifier: ' + str(tree_mcc) + '\n')
print('Matthews correlation coefficient (MCC) RandomForestClassifier: ' + str(random_mcc) + '\n')
print('Classification report SVM: ' + '\n' + str(svm_classreport) + '\n')
print('Confusion matrix SVM: ' + '\n' + str(svm_confusionm) + '\n')
print('Classification report DecisionTreeClassifier: ' + '\n' + str(tree_classreport) + '\n')
print('Confusion matrix DecisionTreeClassifier: ' + '\n' + str(tree_confusionm) + '\n')
print('Classification report RandomForestClassifier: ' + '\n' + str(random_classreport) + '\n')
print('Confusion matrix RandomForestClassifier: ' + '\n' + str(random_confusionm) + '\n')


#Saving the results in output files:
with open ('../project/results/my_results_svm_tree_random2.txt', 'w') as f:
    f.write('Cross-validation scores for SVC: ' + str(svm_cross_mean)+ '\n')
    f.write('Cross-validation scores for DecisionTreeClassifier: '+ str(tree_score_mean)+ '\n')
    f.write('Cross-validation scores for RandomForestClassifier: '+ str(random_score_mean)+ '\n')
    f.write('Matthews correlation coefficient (MCC) SVM: ' + str(svm_mcc) + '\n')
    f.write('Matthews correlation coefficient (MCC) DecisionTreeClassifier: ' + str(tree_mcc) + '\n')
    f.write('Matthews correlation coefficient (MCC) RandomForestClassifier: ' + str(random_mcc) + '\n')
    f.write('Classification report SVM: ' + '\n' + str(svm_classreport) + '\n')
    f.write('Confusion matrix SVM: ' + '\n' + str(svm_confusionm) + '\n')
    f.write('Classification report DecisionTreeClassifier: ' + '\n' + str(tree_classreport) + '\n')
    f.write('Confusion matrix DecisionTreeClassifier: ' + '\n' + str(tree_confusionm) + '\n')
    f.write('Classification report RandomForestClassifier: ' + '\n' + str(random_classreport) + '\n')
    f.write('Confusion matrix RandomForestClassifier: ' + '\n' + str(random_confusionm) + '\n')
f.close()
