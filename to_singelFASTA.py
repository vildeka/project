import os

def parse_singel_fasta(filename, directory_path):
    seq_dict = {}
    with open(filename, 'r') as f:
        f2 = [l for l in (line.strip() for line in f) if l]
        for x, line in enumerate(f2):
            if line[0] == ">":
                key = line[1:]
                #print(key)
            elif x % 3 == 1:
                A = line.strip("\n")
                seq_dict[key] = A
            elif x % 3 == 2:
                #print(x)
                pass
    #print (seq_dict)

    cmd = "mkdir " + directory_path
    print(cmd)
    os.system(cmd)
    
    #for k, v in sorted(seq_dict.items()):
    #for k, v in seq_dict.items():
    for k in seq_dict:
        print(k)
        with open(os.path.join(directory_path, '{}.fasta'.format(k)), 'w') as writefile:
            writefile.write('>' + str(k) + '\n'+ str(seq_dict[k]))
            #writefile.write(str(seq_dict[k]))
            print(str(seq_dict[k]))


if __name__ == '__main__':
    result_FASTA = parse_fasta('../project/datasets/PSSM_files/prediction_file.fasta', '../project/datasets/PSSM_files/new_FASTAs/')
    #result_FASTA = parse_fasta('../project/datasets/data.txt', '../project/datasets/PSSM_files/singel_FASTAs/')
    #print (result_FASTA)
