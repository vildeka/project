import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import pylab as plt


from dataparser import parse_fasta, inputvector_X, inputvector_y

#X = train_inputvector_X(parse_fasta('../project/datasets/data.txt'))
#y = inputvector_y(parse_fasta('../project/datasets/data.txt'), )
x_data = parse_fasta('../project/datasets/datamini.txt')
y = inputvector_y(parse_fasta('../project/datasets/datamini.txt'))

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

print()
print()
print (result_average)

#saves score and plot in respective files:
writefile = open('../project/results/crossval_output.txt', 'w')
for k, v in sorted(result_average.items()):
    writefile.write(str(k))
    writefile.write(':')
    writefile.write(str(v) + "\n")
plt.savefig('../project/results/crossval_score_plt.png')
#plt.show()


# n_jobs=-1, -1 means that you are using all awailable cpu.
# cv=5 means fivefold corss validation.

'''{3: array([ 0.5290764 ,  0.54503991,  0.52054795,  0.54857143,  0.54971429]),
5: array([ 0.54960091,  0.58380844,  0.52054795,  0.57028571,  0.57714286]),
7: array([ 0.55758267,  0.59064994,  0.52511416,  0.576     ,  0.59314286]),
9: array([ 0.54732041,  0.61345496,  0.52853881,  0.576     ,  0.58857143]),
11: array([ 0.54618016,  0.61459521,  0.51940639,  0.56228571,  0.57714286]),
13: array([ 0.54503991,  0.60889396,  0.51826484,  0.552     ,  0.57942857]),
15: array([ 0.54161916,  0.61231471,  0.51255708,  0.54742857,  0.58057143]),
17: array([ 0.5336374 ,  0.61687571,  0.51598174,  0.544     ,  0.576     ]),
19: array([ 0.51653364,  0.6054732 ,  0.50342466,  0.53828571,  0.55771429]),
21: array([ 0.51539339,  0.59749145,  0.49429224,  0.54285714,  0.552     ]),
23: array([ 0.50627138,  0.58152794,  0.48858447,  0.53371429,  0.53714286]),
25: array([ 0.50171038,  0.55872292,  0.48630137,  0.52571429,  0.51085714]),
27: array([ 0.48574686,  0.54047891,  0.4760274 ,  0.50857143,  0.49257143]),
29: array([ 0.47890536,  0.50969213,  0.46689498,  0.50057143,  0.47428571]),
31: array([ 0.47206385,  0.49258837,  0.46118721,  0.49028571,  0.46514286])}'''
