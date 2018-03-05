import pandas as pd
from sklearn import svm
#from sklearn.model_selection import train_test_split
import pickle

from FASTA_to_inputvectors import inputvector_X, parse_fasta


# load the model from disk
model = pickle.load(open("first_model.sav", 'rb'))

#converting predicted states from int to str:
def output_to_topo(result_X):
    topo = {1:'H', 2:'S', 3:'C'}
    topology = []
    for element in result:
        topology.append(topo[element])

        str_topo = "".join(topology)
    return str_topo

#prediction:
result_X = []
#dictionary of identifiers and sequences:
prot_dict = parse_fasta("FASTAfile.txt")

for key in prot_dict:
    print(key)
    print(prot_dict[key])
    x_input = list(FASTA_to_Xvector(prot_dict[key]))
    result = model.predict(x_input)
    result_X.append(result)
    topo = output_to_topo(result_X)
    print(topo)




#l = list['CSSSSSCCCCCCHHHHHHHHHHHCCCSSSSSCCHHHHHH']

#l = list['CCHHHHHHHHHCCCCCCCCCCCCCSSSCCCSSS']

#l = list['CSSSSSSCCCHHHHHHHHHHHHCCCCSSSSSSSCC']

if __name__ == '__main__':
    result_svm = output_to_topo(result)

