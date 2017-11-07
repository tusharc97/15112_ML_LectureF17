import struct, os, gzip
import numpy as np
import time
from sklearn.neural_network import MLPClassifier

# GET DATA

''' MNIST database taken from http://yann.lecun.com/exdb/mnist.
Download the four .gz files (found at the top of the webpage) and save them in
a single folder. '''

DATAFILE = 'mnist' # Name of folder the .gz files are in.

# FILE READING

''' Adapted from:
https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
Takes an MNIST .gz file and returns the information in nparray format.'''
def unzip(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

''' Takes the folder that contains the MNIST .gz files and returns nparrays
corresponding to each file.'''
def readMNIST(datafile):
    trainData = unzip(os.path.join(datafile,'train-images-idx3-ubyte.gz'))
    trainLabels = unzip(os.path.join(datafile,'train-labels-idx1-ubyte.gz'))
    testData = unzip(os.path.join(datafile,'t10k-images-idx3-ubyte.gz'))
    testLabels = unzip(os.path.join(datafile,'t10k-labels-idx1-ubyte.gz'))
    return trainData,trainLabels,testData,testLabels

# PREPROCESSING

'''Takes an nparray representing MNIST images and returns the nparray containing
each flattened image (row-major). The resulant nparray is 2D.'''
def flatten(data):
    return np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2]))

'''Takes an nparray representing MNIST labels and returns the nparray containing
each unrolled label. The ith element of the returned nparray is an nparray
containing 10 elements - one 1 at the index corresponding to the value of the
ith element of the input array and 0's everywhere else.'''
def unpack(data):
    newdata = np.zeros((data.shape[0],10))
    for i in range(len(data)):
        newdata[i][data[i]]=1
    return newdata

# NN SETUP

'''Returns the neural network setup.'''
def nn():
    network=MLPClassifier(hidden_layer_sizes=(392,196,98,49),activation='logistic')
    return network

'''Takes in an nparray where each element is an nparray where the ith element is
the probability that the image is a digit i, and returns the nparray of digits
where each element is the digit with the highest probability for each image.'''
def makeChoice(probs):
    return np.argmax(probs, axis = 1)


'''Takes in the predictions and ground truths, and returns (in percentage terms)
the ratio of the number of correctly-predicted images to the total number of
images predicted on.'''
def getAccuracy(pred,real):
    numReal = 0
    for i in range(len(pred)):
        if np.array_equal(pred[i],real[i]): numReal += 1
    return (numReal/len(pred))*100

# MAIN

'''Runs the network on the training and test sets, and returns the network,
training and test accuracies after fitting.'''
def mnist():
    # Get data from .gz files we downloaded.
    start = time.time()
    print("Reading and processing data...")
    trainData,trainLabels,testData,testLabels = readMNIST(DATAFILE)
    trainData,testData = flatten(trainData),flatten(testData)
    trainLabels,testLabels = unpack(trainLabels),unpack(testLabels)
    print("    Done! Time taken = %0.3f s."%(time.time()-start))
    # Fit the network - it's one line!
    start = time.time()
    print("Training network...")
    network = nn()
    network.fit(trainData,trainLabels)
    print("    Done! Time taken = %0.3f s."%(time.time()-start))
    # Run network on training set to find training set accuracy after fitting.
    start = time.time()
    print("Running network on training set...")
    probstrain = network.predict_log_proba(trainData)
    predtrain = unpack(makeChoice(probstrain))
    acctrain = getAccuracy(predtrain,trainLabels)
    print("    Done! Time taken = %0.3f s."%(time.time()-start))
    # Run network on test set to find test set accuracy after fitting.
    start = time.time()
    print("Running network on test set...")
    probstest = network.predict_log_proba(testData)
    predtest = unpack(makeChoice(probstest))
    acctest = getAccuracy(predtest,testLabels)
    print("    Done! Time taken = %0.3f s."%(time.time()-start))
    # Return accuracies.
    return network,acctrain,acctest

if __name__=="__main__":
    network,acctrain,acctest = mnist()
    print("Network accuracy:")
    print('    Training set accuracy: %0.2f%%.'%acctrain)
    print('    Test set accuracy: %0.2f%%.'%acctest)
