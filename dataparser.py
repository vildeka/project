import pandas as pd
def parse_fasta(filename):
    dictionary = {}
 #write FAST into a three lists, header, sequence and topology. Removes \n and >  
    with open(filename, 'r') as f:
        for x, line in enumerate(f):
            if line[0] == ">":
                key = line[1:-1] #alternative to remove both ">" and "\n"
            elif x % 3 == 1: # gives every second line (% called modulus %3)
                A = line.strip("\n")
            elif x % 3 == 2: # gives every third (line ==0 indicates starting at first line. ==2 indicates starting at third line)
                B = line.strip("\n")
                dictionary[key] = [A, B]

    print (dictionary)
 
#makes the dictionary into a panda     
    df = pd.DataFrame(data=dictionary)
    return (df)
  
        
if __name__ == '__main__':    
    #result_FASTA = parse_fasta("FASTAfile.txt")
    #print (result_FASTA)
    
    result_FASTA = parse_fasta("datamini.txt")
    print (result_FASTA)            


