import os

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
    #print (seq_dict)

    cmd = "mkdir ../project/datasets/PSSM_files/singel_FASTAs/"
    #cmd2 = "cd datasets/PSSM_FASTA/"
    print(cmd)
    os.system(cmd)
    #print(cmd2)

    #os.system(cmd2)

    for k, v in seq_dict.items():
        with open(os.path.join('../project/datasets/PSSM_files/singel_FASTAs/', '{}fasta'.format(k)), 'w') as writefile:
            writefile.write('>' + str(k) + '\n')
            writefile.write(str(v))



if __name__ == '__main__':
    result_FASTA = parse_fasta('../project/datasets/FASTAfilemini.fasta')
    #print (result_FASTA)
