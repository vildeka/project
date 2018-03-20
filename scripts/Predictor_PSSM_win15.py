from sklearn import svm
#from sklearn.model_selection import train_test_split
import pickle
import sys
import os
import glob

from PSSM_parser import parse_fasta, PSSM_parser, PSSM_X_vector
'''info. something is wrong to run predictor you have to specify the filename and directory also in the PSSM_parser.py file. it is not reading the input rom the terminal for some reason'''
lenght = len(sys.argv)
#print(lenght)
if lenght == 1:
    print ('please specify FASTA with path as argv[1] and')
    print ('path to directory where corresponding PSSM files are stored as argv[2]')
    print ('if you wish to use a test file specify: ../datasets/PSSM_files/test_PSSM/PSSM_test.fasta ../datasets/PSSM_files/test_PSSM/')
    exit(1)
elif lenght == 2:
    input_file = sys.argv[1]
    directory = sys.argv[2]


# load the model from disk
model = pickle.load(open('../models/PSSM_win15_model.sav', 'rb'))

#converting predicted states from int to str:
def output_to_topo(result_X):
    topo = {1:'H', 2:'S', 3:'C'}
    topology = []
    for element in result_X:
        topology.append(topo[element])
    str_topo = "".join(topology)
    return str_topo

#prediction:
result_X = []
#dictionary of identifiers and sequences:


def multiple_seq_inputvector_X(input_file, directory):
    input_file = sys.argv[1]
    directory = sys.argv[2]
    parse_dict = parse_fasta(input_file)
    out_file = '../results/PSSM_prediction_test_output.fasta'
    writefile = open(out_file, "w")
    for filename in os.listdir(directory):
        if filename.endswith(".pssm"):
            os.path.join(directory, filename)
            file_pssm = os.path.join(directory, filename)
            #print(file_pssm)
            print('>'+filename[:-11])
            key = parse_dict[filename[:-11]]
            print(key[0])
            dict_values = parse_dict[filename[:-11]]
            #print(dict_values[0])
            X_input = PSSM_X_vector(PSSM_parser(file_pssm))
            #print(X_input.shape)
            #X_input.extend()
            result = model.predict(X_input)
            result_X.append(result)
            topo = output_to_topo(result)
            print(topo)

            writefile.write(str(filename[:-11]) + "\n")
            writefile.write(str(key[0]) + "\n")
            writefile.write(topo + "\n")
        else:
            pass
    writefile.close()
    print ()
    print ('result is to be found in the path:', out_file)
    print ()
if __name__ == '__main__':
    output_to_topo(result_X)
    multiple_seq_inputvector_X(sys.argv[1], sys.argv[2])
    #train_inputvector_X(input_file, directory)
    #inputvector_y(input_file, directory)
    #multiple_seq_inputvector_X('../datasets/test_data0.3.txt', '../datasets/PSSM_files/test_0.3/')
    #multiple_seq_inputvector_X('../datasets/PSSM_files/test_PSSM/PSSM_test.fasta', '../datasets/PSSM_files/test_PSSM/')
