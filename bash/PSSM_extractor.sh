#!/bin/bash
#echo This bash creates PSSM of FASTA file

#python3 -c 'import to_singelFASTA.py; printparse_fasta(filename, directory_path)

#makeblastdb -in /home/u2361/project/datasets/PSSM_files/uniprot_sprot.fasta -dbtype prot -out ../project/datasets/PSSM_files/DB_SWISSPROT/swissprot_db



for entry in ../datasets/PSSM_files/singel_FASTAs/*.fasta
do psiblast -query $entry -evalue 0.01 -db ../datasets/PSSM_files/DB_SWISSPROT/swissprot_db -num_iterations 3 -out_ascii_pssm $entry.pssm -num_threads=8
done
echo fin
