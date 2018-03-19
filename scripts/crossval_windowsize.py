import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import os


from dataparser import parse_fasta, inputvector_X, inputvector_y

X = train_inputvector_X(parse_fasta('../project/datasets/data.txt'))
y = inputvector_y(parse_fasta('../project/datasets/data.txt'), )
#x_data = parse_fasta('../project/datasets/datamini.txt')
#y = inputvector_y(parse_fasta('../project/datasets/datamini.txt'))

results = dict()
result_average = dict()

for windowsize in range(3, 20, 2):
    #list for one protein
    X = list()

    for sequence, topology in x_data.values():
        X_temp = inputvector_X(sequence, windowsize)
        X.append(X_temp)

    X = np.concatenate(X)
    model = svm.SVC(decision_function_shape='ovo')
    score = cross_val_score(model, X, y, cv=5, verbose=True, n_jobs=-1)
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


#cmd = "mkdir ../project/results"
#print(cmd)
#os.system(cmd)

#saves score and plot in respective files:
writefile = open('../project/results/crossval_output.txt', 'w')
for k, v in sorted(result_average.items()):
    writefile.write(str(k))
    writefile.write(':')
    writefile.write(str(v) + "\n")
plt.savefig('../project/results/test_plt.png')
#plt.show()


# n_jobs=-1, -1 means that you are using all awailable cpu.
# cv=5 means fivefold corss validation.
