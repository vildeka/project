# Project for MTLS
Make a 3-state predictor. 


*Script description*
**dataparser.py**
This cointain four functions:
`def parse_fasta(filename):`
#writes the dataset for traning into a dictionary called `parse_dict` with id as key and [sequence, topology] as value. Removes \n and >
`def inputvector_X(sequence, window=7):`
#Takes one string of amino acids at a time, and creates windows that are encoded into binary and finaly converted to numpy array. The shape is in form of (window, nr of windows) where (row, colum) 
#Each numpy array representing one protein 
`def train_inputvector_X(parse_dict):`
#loopes over the dictionary `parse_dict` and extracts only the sequence to feeds it into the `def inputvector_X(sequence, window=7):`. each of the numpy arrays created from that function extended into a list called X_vector, so that you get a list of arrays. this list is finaly created into a numpy array.
#this is the final X_vector that is used for traning 
`def inputvector_y(parse_dict):`
#loopes over the dictionary `parse_dict` and extracts only the topology and encodes it into intergears. Appends them inot a list calle y_vector and turns it into a numpy array. 
#this is the final y_vector that is used for traning 


**train_test.py**
#imports the X_vector and the y_vector from the dataparser.py file
#trains the model with deafult parameter except for 'one vs one'
#saves the model with pickle 

**cross_validation**
#imports the X_vector and the y_vector from the dataparser.py file
#for loop, that does fivefold cross validation for every second windowsizes in range 3-20
#stores the result in dictionary. With windowsize as key and the 5 scores as value. 
 
**FASTA_to_inputvector.py**
`def parse_fasta(filename):`
#write FASTA file for prediction into a dictionary called `seq_dict` with id as key and [sequence] as value. Removes \n and >
`def inputvector_X(sequence, window=7):`
#is the same function as in dataparser.py. 

**svm_prediction.py**
#imports the function inputvector_X() and parse_fasta() from FASTA_to_inputvector
#loades in the models saved with pickle
#runs the prediction on the X_vector created by inputvector_X()
#Converts the prediction from intergears to string (H, S, C).

*Datset description*
I have 3 training datsets in my datset folder: data.txt, datamini.txt, dataminix2.txt 
**data.txt** 
is my complete datset and contains 399 sequences. I do not recomend running that. Therfore I have created two additional datsets. 
**datamini.txt** 
is the one I have used for training the model and doing the cross-validation. 
**dataminix2.txt**
The last dataset called dataminix2.txt is very small containing only 3 sequences. It was used to for easy readabilety when making the scripts. 


