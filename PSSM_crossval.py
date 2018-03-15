import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import os

from PSSM_parser import parse_fasta, PSSM_parser, PSSM_X_vector, train_inputvector_X, inputvector_y


X, topo = train_inputvector_X('../project/datasets/PSSM_files/test_PSSM/PSSM_test.fasta', '../project/datasets/PSSM_files/test_PSSM')
y = inputvector_y('datasets/PSSM_files/test_PSSM/PSSM_test.fasta', 'datasets/PSSM_files/test_PSSM')

#X, topo = train_inputvector('../project/datasets/train_data0.7.txt', '../project/datasets/PSSM_files/train_0.7')
#y = inputvector_y('../project/datasets/train_data0.7.txt', '../project/datasets/PSSM_files/train_0.7')

#uses sklearn module to generate train and test sets by splitting original dataset:
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

print(len(y))
print(X.shape)


results = dict()
result_average = dict()

for windowsize in range(3, 20, 2):    
    #model = svm.SVC(decision_function_shape='ovo')
    #score = cross_val_score(model, X, y, cv=5, verbose=True, n_jobs=-1)
    model = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    score = cross_val_score(model, X, y, cv = 10, verbose=True, n_jobs=-1)
    results[windowsize] = score
    

for window, scores in results.items():
    score = np.average(scores)
    result_average[window] = score

    plt.errorbar(window, score, color='b', yerr=np.std(scores), marker='o')
    plt.title('Score of Window size')
    plt.xlabel('Window Size')
    plt.ylabel('Score')
print()
print()
print (result_average)


#cmd = "mkdir ../project/cool"
#print(cmd)
#os.system(cmd)

#saves score and plot in respective files:
writefile = open('../project/results/PSSM_crossval_output.txt', 'w')
for k, v in sorted(result_average.items()):
    writefile.write(str(k))
    writefile.write(':')
    writefile.write(str(v) + "\n")
plt.savefig('../project/results/PSSM_crossval_score_plt.png')
#plt.show()

'''
# n_jobs=-1, -1 means that you are using all awailable cpu.
# cv=5 means fivefold corss validation.

#if __name__ == '__main__':    
    #result_y = multiple_seq_inputvector_X('../project/datasets/train_data0.7.txt', '../project/datasets/PSSM_files/train_0.7')
    #print (result_y)'''
