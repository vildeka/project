#===================parser========================
import pandas as pd
import numpy as np
def parse_fasta(filename):
    parse_dict = {}
    #sequence = ""
 #write FAST into a three lists, header, sequence and topology. Removes \n and >
    with open(filename, 'r') as f:
        for x, line in enumerate(f):
            if line[0] == ">":
                key = line[1:-1] #alternative to remove both ">" and "\n"
            elif x % 3 == 1: # gives every second line (% called modulus %3)
                A = line.strip("\n")
                #sequence.append(line.strip("\n"))
            elif x % 3 == 2: # gives every third (line ==0 indicates starting at first line. ==2 indicates starting at third line)
                B = line.strip("\n")
                #newA = [for i in A ]
                parse_dict[key] = [A, B]

    #print (parse_dict)

#makes the dictionary into a panda
    df = pd.DataFrame.from_dict(parse_dict, orient = 'index')
    df.columns = ['sequence', 'topology']
    return (parse_dict)

#make panda into numpy: df.values

#=========================input vectors (topology)==============
def input_topo(parse_dict):
    protein = []

    topo_dict = parse_fasta("dataminix2.txt")
    #val = topo_dict.values()
    topo = {'H':1, 'S':2, 'C':3}
    #print(val)

    for val in topo_dict.values():
        topology = np.empty((0,), int)
        #print(topology)
        for s in val[1]:
            topology = np.append(topology, np.array([[topo[s]]], axis=0)
        #protein.append(topology)
    print(topology)

#https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
# https://stackoverflow.com/questions/22392497/how-to-add-a-new-row-to-an-empty-numpy-array
'''
    print("------------------------------------------------------------")
    for val in parse_dict.values():
            for l in val:
                if l == H:
                    topology.append(1)
                elif l == S:
                    topology.append(2)
                else:
                    topology.append(3)
        print (val[1])'''
#=========================input vectors (words)==============
'''
def input_words(parse_dict):

    vals = np.identity(20, dtype=int)
    keys = list("ACDEFGHIKLMNPQRSTVWY")
    #print(keys)
    aa_dict = dict(zip(keys, vals.T))
    aa_dict['0'] = 'array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])'
    #print(aa_dict)

    window = 3
    bla = 1
    bla2 = 2
    for val in parse_dict.values():
        for i in range(len(val)):
            print (i)
            words[sequence[i:window+i:]] = aa_dict[sequence[i]], aa_dict[sequence[bla-i]], aa_dict[sequence[bla2-i]]
    print (words)'''
#=========================One-hot encoding==============
# useless shit:
'''from numpy import argmax
def one_hot_encoding(filename):
    df = parse_fasta(filename)
    one_hot = pd.get_dummies(df['sequence'])
    print(one_hot)
    df = df.drop('sequence', axis=1)
    df.join(one_hot)
    print(df)

    for column in df:
        if column == df.loc[0]:
            print("cool")

    aa = "ACDEFGHIKLMNPQRSTVWY"
    d
    aa_to_int = dict((j, i) for i, j in enumerate(aa))
    int_to_aa = dict((i, j) for i, j in enumerate(aa))
    print (aa_to_int)
    #integer_encoded = [aa_to_int[aa] for aa in data]
    #print(integer_encoded)
'''
#https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
#http://queirozf.com/entries/scikit-learn-pipeline-examples
# http://queirozf.com/entries/pandas-dataframe-by-example#select-rows-by-index-value
#=======================================================
# Import `train_test_split`
# from sklearn.cross_validation import train_test_split

# Split the `digits` data into training and test sets
#X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)


if __name__ == '__main__':
    result_FASTA = parse_fasta("datamini.txt")
    print (result_FASTA)

    result_topo = input_topo("parse_fasta(filename)")

    #result_words = input_words("parse_fasta(filename)")
