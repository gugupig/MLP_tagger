# MLP_tagger
An implementation of a multilayer percetron using Numpy (POS tagger)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~Please use Train.py to see the prediction of our two existing model~
~Plase use softmax_classifier.py to see the result of a linear model~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to re-train the model from scratch, please uncomment the necessary lines

This is a short introductions, if you want to know the details about how each sub-programme function, please read the comments inside each script

NN_Vectorization: Neural network class, you will find detailed descriptions 
for every function used in our training and at the bottom of the script, there is a toy XOR example to test the network

Data_Prep: Preparing training data, however, you don't need to run it because all datas have been already extracted and stored in :
examples_dev.pkl
examples_test.pkl
examples_train.pkl
W2V: convert word into word vectors, you don't need to run it either
model_10.bin:word embedding model trained on our corpus by using libray Gensin.Word2Vec (no pre-trained models are used)

Mini_batch GD_1ST and CLR_2ND are our two trained model who yield 
the best result, you can load them by using Data_Prep.NN_load(path)
they are basically two objects of NN_Vec class, please read the comments in NN_Vectorization to know there attributes and methods

Mini-batch GD is trained by using mini-batch gradient descent
CLR is trained by using the Cyclical Learning Rate technique
Please read our report for more infomations
