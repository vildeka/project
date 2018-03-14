import os
import numpy as np
#===================fasta parser========================
def parse_fasta(filename):
#'''write FAST into a dictionary with id as key and [sequence, topology] as value. Removes \n and >'''
    parse_dict = {}
    with open(filename, 'r') as f:
        f2 = [l for l in (line.strip() for line in f) if l]
        for x, line in enumerate(f2):
            if line[0] == ">":
                key = line[1:]
            elif x % 3 == 1:
                A = line.strip("\n")
            elif x % 3 == 2:
                B = line.strip("\n")
                parse_dict[key] = [A, B]
    #print(parse_dict)
    return parse_dict


#===================PSSM parser========================
def PSSM_parser(PSSM_file):
    PSSM_format = (np.genfromtxt(PSSM_file,
                                 skip_header = 3,
                                 skip_footer = 5,
                                 autostrip = True,
                                 usecols = range(22, 42)))/100
    return PSSM_format
#================input vectors words/features==============
def PSSM_X_vector(PSSM_format, window=15):
    padding = window//2
    #final input:
    features=[]

    for i in range(len(PSSM_format)):

        if i > padding and i < len(PSSM_format) - padding - 1:
            #-1 because you want second to last element
            PSSM_seq = PSSM_format[i-padding:i+padding+1]
            #+1 is to get the last element as well [icluded:notincluded]
            temp_merge=[]
            for lists in PSSM_seq:#merging list by extending
                temp_merge.extend(lists)
            features.append(temp_merge)
        elif i <= padding:
            # head
            this_PSSM = PSSM_format[:i + padding + 1]
            zeros_needed = window - len(this_PSSM)
            PSSM_seq = [0] *(20*zeros_needed)
            for lists in this_PSSM:
                PSSM_seq.extend(lists)
            features.append(PSSM_seq)
        else:
            # tail
            this_PSSM = PSSM_format[i-1:]
            zeros_needed = window - len(this_PSSM)
            PSSM_seq=[]
            for lists in this_PSSM:
                PSSM_seq.extend(lists)
            PSSM_seq.extend([0] * (20*zeros_needed))
            features.append(PSSM_seq)

    return np.array(features)
    #print (np.array(features.shape)

def train_inputvector_X(directory):
    parse_dict = parse_fasta('datasets/PSSM_files/test_PSSM/PSSM_test.fasta')
    X_input = []
    top=[]
    #directory = "datasets/PSSM_files/singel_FASTAs"
    for filename in os.listdir(directory):
        if filename.endswith(".pssm"):
            os.path.join(directory, filename)
            file_pssm = os.path.join(directory, filename)
            #print(file_pssm)
            feature = PSSM_X_vector(PSSM_parser(file_pssm))
            dict_value = parse_dict[filename[:-11]]
            top.extend(dict_value[1])
    return np.array(X_input), top

#================input vector topology==============
def inputvector_y(topology):
    # dictionary of topology:
    topo = {'H':1, 'S':2, 'C':3}

    #final input:
    y_input = []
    features, topology = train_inputvector_X("datasets/PSSM_files/test_PSSM")
    for i in range(len(topology)):
        y_input.append(topo[topology[i]])

    return np.array(y_input)



if __name__ == '__main__':
    #result_parser = parse_fasta()

    result_multiple_seq = train_inputvector_X("datasets/PSSM_files/test_PSSM")

    result_input_y = inputvector_y(parse_fasta('datasets/PSSM_files/test_PSSM/PSSM_test.fasta'))
