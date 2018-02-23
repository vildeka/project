#===================parser========================
import pandas as pd
def parse_fasta(filename):
    dictionary = {}
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
                dictionary[key] = [A, B]

    #print (dictionary)

#makes the dictionary into a panda
    df = pd.DataFrame.from_dict(dictionary, orient = 'index')
    df.columns = ['sequence', 'topology']
    return (df)

    #sequence = df.loc[0]
    #return(sequence)
#make panda into numpy: df.values

#=========================One-hot encoding==============
from numpy import argmax
def one_hot_encoding(filename):
    df = parse_fasta(filename)
    one_hot = pd.get_dummies(df['sequence'])
    print(one_hot)
    df = df.drop('sequence', axis=1)
    df.join(one_hot)
    print(df)

    '''for column in df:
        if column == df.loc[0]:
            print("cool")

    aa = "ACDEFGHIKLMNPQRSTVWY"

    aa_to_int = dict((j, i) for i, j in enumerate(aa))
    int_to_aa = dict((i, j) for i, j in enumerate(aa))
    print (aa_to_int)
    #integer_encoded = [aa_to_int[aa] for aa in data]
    #print(integer_encoded)'''

#https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
#http://queirozf.com/entries/scikit-learn-pipeline-examples
# http://queirozf.com/entries/pandas-dataframe-by-example#select-rows-by-index-value
#=======================================================
# Import `train_test_split`
from sklearn.cross_validation import train_test_split

# Split the `digits` data into training and test sets
#X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)


if __name__ == '__main__':
    result_FASTA = parse_fasta("datamini.txt")
    print (result_FASTA)

    result_ecoding = one_hot_encoding("datamini.txt")
