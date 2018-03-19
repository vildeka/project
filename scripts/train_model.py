from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
import pickle

from dataparser import parse_fasta, train_inputvector_X, inputvector_y

def train_model(filename):

    X = train_inputvector_X(parse_fasta(filename))
    y = inputvector_y(parse_fasta(filename))

    print(X.shape)
    #create model and give other than dtype:
    #model = svm.SVC(decision_function_shape='ovo')
    #training data:
    model.fit(X, y)
    accuracy = model.score(X, y)
    print(accuracy)

    #using pickle to save model to disk:
    filename = '../project/models/mini_win15_model.sav'
    pickle.dump(model, open(filename, 'wb'))


if __name__ == '__main__':
    train_model('../project/datasets/train_data0.7.txt')
    #train_model('../project/datasets/data.txt')
