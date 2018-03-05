import numpy as np


def parse_fasta(filename):
    seq_dict = {}
    with open(filename, 'r') as f:
        f2 = [l for l in (line.strip() for line in f) if l]
        for x, line in enumerate(f2):
            if line[0] == ">":
                key = line[1:-1]
            else:
                A = line.strip("\n")
                seq_dict[key] = A    
    return(seq_dict)
    

def inputvector_X(sequence, window=3): #identical to def inputvector_X(sequence, window=3):
    padding = window//2
    
    # dictionary of aa:
    vals = np.identity(20, dtype=int)
    keys = list("ACDEFGHIKLMNPQRSTVWY")
    aa_dict = dict(zip(keys, vals.T))
    aa_dict['0'] = np.zeros(20, dtype=int)
        
    
    #final input:
    features = []
    #list of words:   
    word_seq = []
    
    for i in range(len(sequence)):
        if i > padding and i < len(sequence) - padding - 1:
            #-1 because you want second to last element 
            word_seq.append(sequence[i-padding:i+padding+1])
            #+1 is to get the last element as well [icluded:notincluded]
        elif i <= padding:
            # head
            this_word = sequence[:i + padding + 1] #[:2] = PT, [:3]= PTV
            zeros_needed = window - len(this_word)
            #print(this_word)
            #print('0'*zeros_needed)
            word_seq.append('0' * zeros_needed + this_word)
        else:
            # tail
            this_word = sequence[i-1:] #[43-1:]= ASC, [44-1]= SC
            zeros_needed = window - len(this_word)
            word_seq.append(this_word+'0' * zeros_needed)
        
    for word in word_seq:    
        this_word = list(map(lambda n: aa_dict[n], word))    
        features.append(np.concatenate(this_word))
    return np.array(features)

    


if __name__ == '__main__':
    result_FASTA = parse_fasta("FASTAfile.txt")
    print (result_FASTA)
    
    result_FASTA = FASTA_to_inputvectorX(parse_fasta("FASTAfile.txt"))
    #print (result_FASTA)
