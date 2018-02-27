import pandas as pd
from sklearn import svm
#from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle

from FASTA_to_inputvectorX import FASTA_to_inputvectorX, parse_fasta


# load the model from disk
model = pickle.load(open("first_model.sav", 'rb'))


#prediction:
result = model.predict(FASTA_to_inputvectorX(parse_fasta("FASTAfile.txt")))
print(result)

topo = {1:'H', 2:'S', 3:'C'}
topology = []
for element in result:
    topology.append(topo[element])

print(topology)

l = list['CSSSSSCCCCCCHHHHHHHHHHHCCCSSSSSCCHHHHHH']

l = list['CCHHHHHHHHHCCCCCCCCCCCCCSSSCCCSSS']

l = list['CSSSSSSCCCHHHHHHHHHHHHCCCCSSSSSSSCC']
