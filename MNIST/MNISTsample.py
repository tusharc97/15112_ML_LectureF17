import struct, os, gzip
import numpy as np
import time
from sklearn.neural_network import MLPClassifier

'''
GET DATA
'''

# MNIST database taken from http://yann.lecun.com/exdb/mnist.
# Download the four .gz files (found at the top of the webpage) and save them in
# a single folder.

DATAFILE = 'mnist' # Name of folder the .gz files are in.

'''
FILE READING
'''

# adapted https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
def unzip(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def readMNIST(datafile):
    trainData = unzip(os.path.join(datafile,'train-images-idx3-ubyte.gz'))
    trainLabels = unzip(os.path.join(datafile,'train-labels-idx1-ubyte.gz'))
    testData = unzip(os.path.join(datafile,'t10k-images-idx3-ubyte.gz'))
    testLabels = unzip(os.path.join(datafile,'t10k-labels-idx1-ubyte.gz'))
    return trainData,trainLabels,testData,testLabels

'''
PREPROCESSING
'''

def flatten(data):
    return np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2]))

def unpack(data):
    newdata = np.zeros((data.shape[0],10))
    for i in range(len(data)):
        newdata[i][data[i]]=1
    return newdata

'''
NN SETUP
'''

def nn():
    network=MLPClassifier(hidden_layer_sizes=(392,196,98,49),activation='logistic')
    return network

def makeChoice(probs):
    return np.argmax(probs, axis = 1)

def getAccuracy(pred,real):
    numReal = 0
    for i in range(len(pred)):
        if np.array_equal(pred[i],real[i]): numReal += 1
    return (numReal/len(pred))*100

'''
MAIN
'''

def mnist():
    start = time.time()
    print("Reading and processing data... ",)
    trainData,trainLabels,testData,testLabels = readMNIST(DATAFILE)
    trainData,testData = flatten(trainData),flatten(testData)
    trainLabels,testLabels = unpack(trainLabels),unpack(testLabels)
    print("    Done! Time taken = %0.3f s."%(time.time()-start))
    start = time.time()
    print("Training network... ")
    network = nn()
    network.fit(trainData,trainLabels)
    print("    Done! Time taken = %0.3f s."%(time.time()-start))
    start = time.time()
    print("Running network on training set... ")
    probstrain = network.predict_log_proba(trainData)
    predtrain = unpack(makeChoice(probstrain))
    acctrain = getAccuracy(predtrain,trainLabels)
    print("    Done! Time taken = %0.3f s."%(time.time()-start))
    start = time.time()
    print("Running network on test set...")
    probstest = network.predict_log_proba(testData)
    predtest = unpack(makeChoice(probstest))
    acctest = getAccuracy(predtest,testLabels)
    print("    Done! Time taken = %0.3f s."%(time.time()-start))
    return network,acctrain,acctest

if __name__=="__main__":
    network,acctrain,acctest = mnist()
    print("Network accuracy:")
    print('    Training set accuracy: %0.2f%%.'%acctrain)
    print('    Test set accuracy: %0.2f%%.'%acctest)
