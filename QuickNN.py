"""
    Simple genetic algorithm class in Python
    Copyright (C) 2014 Achilleas Koutsou

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


TODO:
    - Build-in help (command-line arguments)
    - Function to pickle dump trained weights
    - Function to load saved weights

"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1./(1+np.exp(-x))

def build_and_init(numin, numh1, numh2, numout):
    nnet = NeuralNet(numin,numh1,numh2,numout)
    return nnet

def forward_pass(nnet,inputs):
    return nnet.calculateOutput(inputs)

def backprop(nnet,inputs,actOut,desOut,learningRate,momentum):
    deltasOutput = actOut*(1-actOut)*(actOut-desOut)
    if nnet.numHidden2 > 0:
        nnet.outputLayer =\
                nnet.outputLayer -\
                learningRate*np.outer(deltasOutput, nnet.layerTwoOutput)\
                + momentum*(nnet.outputLayer - nnet.outputLayerPrev)
        deltasH2 =\
                nnet.layerTwoOutput[:-1]*(1-nnet.layerTwoOutput[:-1])*\
                np.dot(deltasOutput, nnet.outputLayer[:,:-1])
        nnet.hiddenLayerTwo = nnet.hiddenLayerTwo -\
                learningRate*np.outer(deltasH2, nnet.layerOneOutput)\
                + momentum*(nnet.hiddenLayerTwo\
                - nnet.hiddenLayerTwoPrev)

        deltasH1 =\
                nnet.layerOneOutput[:-1]*(1-nnet.layerOneOutput[:-1])*\
                np.dot(deltasH2, nnet.hiddenLayerTwo[:,:-1])
        nnet.hiddenLayerOne = nnet.hiddenLayerOne -\
                learningRate*np.outer(deltasH1, inputs) +\
                momentum*(nnet.hiddenLayerOne - nnet.hiddenLayerOnePrev)
    else:
        deltasH1 = nnet.layerOneOutput[:-1]*(1-nnet.layerOneOutput[:-1])*\
                np.dot(deltasOutput, nnet.outputLayer[:,:-1])
        nnet.hiddenLayerOne = nnet.hiddenLayerOne -\
                learningRate*np.outer(deltasH1,inputs) +\
                momentum*(nnet.hiddenLayerOne - nnet.hiddenLayerOnePrev)

    nnet.hiddenLayerOnePrev = np.copy(nnet.hiddenLayerOne)
    if nnet.numHidden2 > 0:
        nnet.hiddenLayerTwoPrev = np.copy(nnet.hiddenLayerTwo)

    nnet.outputLayerPrev = np.copy(nnet.outputLayer)

    return



class NeuralNet:
    """ The neural network object """
    def __init__(self,numin,numh1,numh2,numout):
        """
        Layers will be defined as arrays such that calling
        self.hiddenLayerOne[0,:] or simply self.hiddenLayerOne[0]
        returns the weights between the input layer and the first hidden neuron
        """
        self.numInputs = numin
        self.numHidden1 = numh1
        self.numHidden2 = numh2
        self.numOutputs = numout

        self.layerOneOutput = np.zeros(self.numHidden1+1) # +1 bias
        self.layerTwoOutput = np.zeros(self.numHidden2+1) # +1 bias
        self.networkOutput = np.zeros(self.numOutputs)

        # initializing weights to randoms within [-0.5, 0.5]
        # this could be a separate function, but who cares
        self.hiddenLayerOne = \
                np.random.rand(self.numHidden1, self.numInputs+1)-0.5 # +1 bias
        if self.numHidden2 > 0:
            self.hiddenLayerTwo = \
                    np.random.rand(self.numHidden2, self.numHidden1+1)-0.5
            self.outputLayer =\
                    np.random.rand(self.numOutputs, self.numHidden2+1)-0.5
        else:
            self.outputLayer =\
                    np.random.rand(self.numOutputs, self.numHidden1+1)-0.5

        # initializing old weights to zero for momentum
        self.hiddenLayerOnePrev = np.zeros(np.shape(self.hiddenLayerOne))
        if self.numHidden2 > 0:
            self.hiddenLayerTwoPrev = np.zeros(np.shape(self.hiddenLayerTwo))
        self.outputLayerPrev = np.zeros(np.shape(self.outputLayer))

    def calculateOutput(self,inputs):
        self.layerOneOutput =\
                sigmoid(np.sum(inputs*self.hiddenLayerOne,axis=1))
        self.layerOneOutput = np.append(self.layerOneOutput, 1) # bias
        if self.numHidden2 > 0:
            self.layerTwoOutput =\
                    sigmoid(np.sum(self.layerOneOutput*self.hiddenLayerTwo,\
                    axis=1))
            self.layerTwoOutput = np.append(self.layerTwoOutput, 1) # bias
            self.networkOutput =\
                    sigmoid(np.sum(self.layerTwoOutput*self.outputLayer,\
                    axis=1))
        else:
            self.networkOutput =\
                    sigmoid(np.sum(self.layerOneOutput*self.outputLayer,\
                    axis=1))

        return self.networkOutput


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Please provide a configuration file.")

    parameter_filename = sys.argv[1]
    if os.path.isfile(parameter_filename):
        parameter_file = open(parameter_filename,'r')
    else:
        sys.exit("Parameter file does not exist!")

    print("Reading parameters from",parameter_filename)


    numHiddenLayerOneNeurons = 0
    numHiddenLayerTwoNeurons = 0
    numInputNeurons = 0
    numOutputNeurons = 0
    learningRate = 0.0
    momentum = 0
    maxIterations = 0
    trainFileName = ""
    testFileName = ""
    delimiter = " "

    # reading parameters
    filelines = parameter_file.readlines()
    for line in filelines:
        words = line.split()
        if words[0] == "numHiddenLayerOneNeurons":
            try:
                numHiddenLayerOneNeurons = int(words[1])
            except ValueError:
                sys.exit("Error in parameter file. Check values!")
        elif words[0] == "numHiddenLayerTwoNeurons":
            try:
                numHiddenLayerTwoNeurons = int(words[1])
            except ValueError:
                sys.exit("Error in parameter file. Check values!")
        elif words[0] == "numInputNeurons":
            try:
                numInputNeurons = int(words[1])
            except ValueError:
                sys.exit("Error in parameter file. Check values!")
        elif words[0] == "numOutputNeurons":
            try:
                numOutputNeurons = int(words[1])
            except ValueError:
                sys.exit("Error in parameter file. Check values!")
        elif words[0] == "learningRate":
            try:
                learningRate = float(words[1])
            except ValueError:
                sys.exit("Error in parameter file. Check values!")
        elif words[0] == "momentum":
            try:
                momentum = float(words[1])
            except ValueError:
                sys.exit("Error in parameter file. Check values!")
        elif words[0] == "maxIterations":
            try:
                maxIterations = int(words[1])
            except ValueError:
                sys.exit("Error in parameter file. Check values!")
        elif words[0] == "trainFile":
            try:
                trainFileName = words[1]
            except ValueError:
                sys.exit("Error in parameter file. Check values!")
        elif words[0] == "testFile":
            try:
                testFileName = words[1]
            except ValueError:
                sys.exit("Error in parameter file. Check values!")
        elif words[0] == "delimiter":
            try:
                delimiter = words[1]
            except ValueError:
                sys.exit("Error in parameter file. Check values!")

    # checking if data files exist
    if os.path.isfile(trainFileName) and os.path.isfile(testFileName):
        trainfile = open(trainFileName,'r')
        testfile = open(testFileName,'r')
    else:
        sys.exit("ERROR: Please check that the data files exist\n")

    traindata = trainfile.readlines()
    testdata = testfile.readlines()

    # checking if configuration matches data
    numValues = len(traindata[0].split(delimiter))

    if numValues != numInputNeurons+numOutputNeurons:
        sys.exit("ERROR: Network architecture does not match data!\n\
Input neurons: %d, Output neurons: %d, Data colums: %d \n" %\
        (numInputNeurons,numOutputNeurons,numValues))


    # printing parameters for user's information
    print("Building network with architecture:",\
            numInputNeurons,"-",numHiddenLayerOneNeurons,"-",\
            numHiddenLayerTwoNeurons,"-",numOutputNeurons)
    print("Learning rate:",learningRate)
    print("Momentum:",momentum)
    print("Number of iterations (epochs):",maxIterations)
    print("Training data file:",trainFileName)
    print("Testing data file:",testFileName)
    print("\n\n")

    # build the network

    nnet = build_and_init(numInputNeurons, numHiddenLayerOneNeurons,\
            numHiddenLayerTwoNeurons,numOutputNeurons)
    # start the learning process
    trainError = []
    testError = []
    succRate = []
    succRatePM = []
    numTrainData = len(traindata)
    numTestData = len(testdata)
    start_time = time.time()
    for epoch in range(maxIterations):
        epochTrainError = 0
        epochTestError = 0
        epochSuccessRate = 0
        epochSuccessRatePM = 0
        progress = 0
        for trainrow in traindata:
            trainrow = trainrow.split(delimiter)
            trainrow = np.array(trainrow,dtype=float)
            inputs = trainrow[0:nnet.numInputs]
            inputs = np.append(inputs, 1) # adding bias to end of inputs
            desiredOutputs = trainrow[-nnet.numOutputs:]
            actualOutputs = forward_pass(nnet, inputs)
            patternError =\
                    np.sum((desiredOutputs - actualOutputs)**2)/numTrainData
            epochTrainError = epochTrainError + patternError
            backprop(nnet,inputs,actualOutputs,desiredOutputs,\
                    learningRate,momentum)
            progress = progress + 100./len(traindata)
            sys.stdout.write("\rTraining progress: %2.f %%" % progress)
            sys.stdout.flush()
        sys.stdout.write("\r"+" "*40+"\r") # clear the line
        sys.stdout.flush()
        trainError.append(epochTrainError)
        progress = 0
        for testrow in testdata:
            testrow = testrow.split(delimiter) # inefficient; consider changing
            testrow = np.array(testrow,dtype=float)
            inputs = testrow[0:nnet.numInputs]
            inputs = np.append(inputs, 1) # adding bias to end of inputs
            desiredOutputs = testrow[-nnet.numOutputs:]
            actualOutputs = forward_pass(nnet,inputs)
            patternError = \
                    np.sum((desiredOutputs - actualOutputs)**2)/numTestData
            epochTestError = epochTestError + patternError
            if nnet.numOutputs > 1:
                desOutNumerical =\
                        int(np.where(desiredOutputs == max(desiredOutputs))[0])
                actOutNumerical\
                        = int(np.where(actualOutputs == max(actualOutputs))[0])
                if desOutNumerical == actOutNumerical:
                    epochSuccessRate = epochSuccessRate + 1./numTestData
                    epochSuccessRatePM = epochSuccessRatePM + 1./numTestData
                elif desOutNumerical+1 >= actOutNumerical and\
                        desOutNumerical-1 <= actOutNumerical:
                    epochSuccessRatePM = epochSuccessRatePM + 0.5/numTestData
            progress = progress + 100./len(testdata)
            sys.stdout.write("\rTesting progress: %2.f %%" % progress)
            sys.stdout.flush()
        sys.stdout.write("\r"+" "*40+"\r") # clear the line
        sys.stdout.flush()
        testError.append(epochTestError)
        if nnet.numOutputs > 1:
            succRate.append(epochSuccessRate)
            succRatePM.append(epochSuccessRatePM)
        time_now = time.time()
        elapsed = time_now - start_time
        per_epoch = elapsed/(epoch+1)
        est_remaining = per_epoch*(maxIterations-epoch-1)
        print("Epoch %d done!" % epoch)
        print("Time elapsed  : %d seconds\nTime remaining: %d seconds" %\
                (elapsed, est_remaining))
        print("--> Training error:\t%f\n--> Testing error :\t%f\n"\
                % (trainError[-1], testError[-1]))
        if nnet.numOutputs > 1:
            print("--> Success rate:\t%f\n\
--> Success rate pm 1:\t%f" % (succRate[-1],succRatePM[-1]))
        print("--")

    print("\n\nFinal feedforward for check")
    lastSuccessRate = 0
    lastSuccessRatePM = 0
    num_final_output = 20
    if (numTestData <= num_final_output):
        report_indices = np.arange(numTestData)
    else:
        report_indices = np.random.permutation(numTestData)[:num_final_output]
    ind = 0
# save all des-act for histogram
    final_errors = []
    progress = 0
    for testrow in testdata:
        testrow = testrow.split(delimiter)
        testrow = np.array(testrow,dtype=float)
        inputs = testrow[0:numInputNeurons]
        inputs = np.append(inputs, 1) # adding bias to end of inputs
        desiredOutputs = testrow[-numOutputNeurons:]
        actualOutputs = forward_pass(nnet,inputs)
        patternError =\
                np.sum((desiredOutputs - actualOutputs)**2)/numTestData # MSE
        final_errors.append(np.sum(np.abs(desiredOutputs - actualOutputs)))
        epochTestError = epochTestError + patternError
        if nnet.numOutputs > 1:
            desOutNumerical =\
                    int(np.where(desiredOutputs == max(desiredOutputs))[0])
            actOutNumerical\
                    = int(np.where(actualOutputs == max(actualOutputs))[0])
            if desOutNumerical+1 == actOutNumerical:
                lastSuccessRate = lastSuccessRate + 1./numTestData
                lastSuccessRatePM = lastSuccessRatePM + 1./numTestData
            elif desOutNumerical+1 >= actOutNumerical\
                    and desOutNumerical-1 <= actOutNumerical:
                lastSuccessRatePM = lastSuccessRatePM + 0.5/numTestData
        ind = ind + 1
        progress = progress + 100./len(testdata)
        sys.stdout.write("\rFinal test progress: %2.f %%" % progress)
        sys.stdout.flush()
    sys.stdout.write("\r"+" "*40+"\r") # clear the line
    sys.stdout.flush()
    if nnet.numOutputs > 1:
        print("Final success rate:",lastSuccessRate)
        print("Final success rate pm:",lastSuccessRatePM)
        plt.subplot(311)
        plt.plot(range(maxIterations),trainError,range(maxIterations),testError)
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.axis([0,maxIterations,0,max([max(testError),max(trainError)])])

        plt.subplot(312)
        plt.plot(range(maxIterations),succRate,range(maxIterations),succRatePM)
        plt.xlabel("Epochs")
        plt.ylabel("Success Rate")
        plt.axis([0,maxIterations,0,1.1])

        plt.subplot(313)
        plt.hist(final_errors, 50)
    else:
        plt.subplot(211)
        plt.plot(range(maxIterations),trainError,range(maxIterations),testError)
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.axis([0,maxIterations,0,max([max(testError),max(trainError)])])

        plt.subplot(212)
        [heights, bins] = np.histogram(final_errors, bins=50)
        norm_heights = np.array(heights, dtype=float)/sum(heights)
        norm_heights = np.append(norm_heights, 0)
        plt.plot(bins, norm_heights)
    plt.show()

    print("ALL DONE")
