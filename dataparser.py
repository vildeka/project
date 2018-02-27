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

    return (parse_dict)

#make panda into numpy: df.values



#================input vectors (words/features and topology)==============

def input_vectors(seq_dict, window=3):
    padding = window//2
    
    # dictionary of aa:
    vals = np.identity(20, dtype=int)
    keys = list("ACDEFGHIKLMNPQRSTVWY")
    aa_dict = dict(zip(keys, vals.T))
    aa_dict['0'] = np.zeros(20, dtype=int)
    
    # dictionary of topology:
    topo = {'H':1, 'S':2, 'C':3}
    
    #list of words:
    word_seq = []
    
    #final input:
    topologies = []
    features = []
    
    for sequence, topology in seq_dict.values(): 
        for i in range(len(sequence)):
            topologies.append(topo[topology[i]])
            if i > padding and i < len(sequence) - padding - 1:
                 #-1 because you want second to last element 
                word_seq.append(sequence[i-padding:i+padding+1])
                  #+1 is to get the last element as well [icluded:notincluded]
            elif i <= padding:
                # head
                this_word = sequence[:i + padding + 1] #[:2] = PT, [:3]= PTV
                zeros_needed = window - len(this_word)
                word_seq.append('0' * zeros_needed + this_word)

            else:
                # tail
                this_word = sequence[i-1:] #[43-1:]= ASC, [44-1]= SC
                zeros_needed = window - len(this_word)
                word_seq.append(this_word+'0' * zeros_needed)


    for word in word_seq:    
        this_word = list(map(lambda n: aa_dict[n], word))    
        features.append(np.concatenate(this_word))

    return np.array(features), np.array(topologies)
      
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
    result_FASTA = parse_fasta("dataminix2.txt")
    print (result_FASTA)


    result_words = input_words(parse_fasta("dataminix2.txt"))
