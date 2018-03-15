from sklearn import svm
#from sklearn.model_selection import train_test_split
import pickle
import sys

from dataparser import inputvector_X, parse_fasta

lenght = len(sys.argv)
if lenght == 1: 
    print ('please specify FASTA with path as argv[1]')
    print ('if you wish to use a test file specify ../project/datasets/test_data50.txt')
    exit(1)
elif lenght == 2:
    input_file = sys.argv[1]
 

# load the model from disk
model = pickle.load(open('../project/models/349_random_win15_model.sav', 'rb'))

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
prot_dict = parse_fasta(input_file)
print(prot_dict)

out_file = '../project/results/prediction_50_output.fasta'
writefile = open(out_file, "w")
#for sequence, topology in prot_dict.values():
for key in prot_dict:    
    values = prot_dict[key]
    x_input = list(inputvector_X(values[0]))
    result = model.predict(x_input)
    result_X.append(result)
    topo = output_to_topo(result_X)
    print('>'+key)
    print(values[0])
    print(topo)
    writefile.write('>'+key + "\n")
    writefile.write(values[0] + "\n")
    writefile.write(topo + "\n")

writefile.close()

print ()
print ('result is to be found in the path:', out_file)
print ()



if __name__ == '__main__':
    result_svm = output_to_topo(result)
