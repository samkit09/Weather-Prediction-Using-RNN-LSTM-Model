# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:57:07 2019

@authors: Sahithi
		  Hanumantha Rao
		  Aishwarya
		  Siva
"""

import numpy as np
from LSTM import LSTM

class RecurrentNeuralNetwork:
    #function to initialize RNN with the parameters
    def __init__ (self, inputData, trueOutput, hiddenLayerSize, learningRate, batchSize):

        #input data for training
        self.inputData = inputData
        #actual output data
        self.trueOutput = trueOutput
        #lstm output dimension
        self.hlSize = hiddenLayerSize
        self.learningRate = learningRate
        #as we cannot pass entire dataset we divide the dataset into batches
        self.batchSize = batchSize
        #input data feature count
        self.inputDimension = inputData.shape[1]
        #output feature count
        self.outputDimension = 1

        #lstm to dense layer connecting weight matrix
        self.denseWeigthMatrix = np.random.random(( self.outputDimension, self.hlSize))
        #matirx to update lstm to dense layer connecting weight matrix
        self.updateDenseWeigthMatrix = np.zeros_like(self.denseWeigthMatrix)

        #an array to store cell states after each data point
        self.cellStateArray = np.zeros((self.batchSize+1, self.hlSize))
        #an array to store final predicted value after each data point
        self.finalPredictedArray = np.zeros((self.batchSize+1, self.outputDimension))
        #an array to store lstm output after each data point
        self.hiddenStateArray = np.zeros((self.batchSize+1, self.hlSize))

        #an array to store forget gate values after each data point
        self.forgetGateArray = np.zeros((self.batchSize+1, self.hlSize))
        #an array to store input gate values after each data point
        self.inputGateArray = np.zeros((self.batchSize+1, self.hlSize))
        #an array to store activation gate values after each data point
        self.activationGateArray = np.zeros((self.batchSize+1, self.hlSize))
        #an array to store output gate values after each data point
        self.outputGateArray = np.zeros((self.batchSize+1, self.hlSize))

        #used in updating the weights of the gates and dense layer
        self.gradientMulFactor = 0.950
        self.deltaMulFactor = 0.10
        self.denseWeightAddition = 1e-9
        #intializing LSTM cell
        self.LSTM = LSTM(self.inputDimension, self.hlSize, self.batchSize, self.learningRate)

    def sig(self, val):
        sigVal = 1 / (1 + np.exp(-val))
        return sigVal

    def derivativeSig(self, val):
        derivative=self.sig(val) * (1 - self.sig(val))
        return derivative

    #function for RNN forward pass
    def forwardPass(self, bacthInputData):

        #for each datapoint call the LSTM forward pass to obtain the cellState and hiddenstate
        for time in range(1, bacthInputData.shape[0]):
            #append previous lstm output and present input arrays to give input to lstm cell
            self.LSTM.inputData = np.hstack((self.hiddenStateArray[time-1], bacthInputData[time]))

            cellStateValue, hiddenStateValue, forgetGateValue, inputGateValue, activationGateValue, outputGateValue = self.LSTM.forwardPass()

            #store the above values into the arrays defined above in RNN
            self.cellStateArray[time] = cellStateValue
            self.hiddenStateArray[time] = hiddenStateValue
            self.forgetGateArray[time] = forgetGateValue
            self.inputGateArray[time] = inputGateValue
            self.activationGateArray[time] = activationGateValue
            self.outputGateArray[time] = outputGateValue
            #multiply the hidden state value with weight matrix and pass it through sigmoid to obtain predicted value
            self.finalPredictedArray[time] = self.sig(np.dot(self.denseWeigthMatrix, hiddenStateValue))
        return self.finalPredictedArray

    #function for RNN backward pass to calculate weight updates for dense layer and LSTM cell
    def backwardPass(self, actualBatchOutputData, batchInputData):
        #error in entire batch
        batchError = 0
        #an array to store delta of cell state value for a data point
        deltaOfCellState = np.zeros(self.hlSize)
        #an array to store delta of hidden state value for a data point
        deltaOfHiddenState = np.zeros(self.hlSize)

        #lstm to dense layer connecting weight matrix weights delta update
        deltaWeightUpdate = np.zeros((self.outputDimension, self.hlSize))
        #total delta of weight for forget gate
        totalDeltaForgetGate = np.zeros((self.hlSize, self.inputDimension+self.hlSize))
        #total delta of weight for input gate
        totalDeltaInputGate = np.zeros((self.hlSize, self.inputDimension+self.hlSize))
        #total delta of weight for activation gate
        totalDeltaActivationGate = np.zeros((self.hlSize, self.inputDimension+self.hlSize))

        totalDeltaOutputGate = np.zeros((self.hlSize, self.inputDimension+self.hlSize))

        #a loop to propagate backward to find delta after each data point
        for time in range(self.batchSize-1, -1, -1):
            #error for each data point
            errorForDataPoint = abs(self.finalPredictedArray[time] - actualBatchOutputData[time])
            #sum each data point error into bacth error
            batchError += errorForDataPoint
            # (error * sigmoid derivative of predicted) * hidden state
            deltaWeightUpdate += np.dot(np.atleast_2d(errorForDataPoint * self.derivativeSig(self.finalPredictedArray[time])),
                                        np.atleast_2d(self.hiddenStateArray[time]))

            #multiply error with RNN weight matrix which is sent to LSTM backward pass
            errorForLSTM = np.dot(errorForDataPoint, self.denseWeigthMatrix)
            #append previous lstm output and present input arrays to give input to lstm cell
            self.LSTM.inputData = np.hstack((self.hiddenStateArray[time-1], batchInputData[time]))
            #update lstm cell state to current time cell state value
            self.LSTM.cellState = self.cellStateArray[time]
            #call LSTM backward pass
            deltaForgetGate, deltaInputGate, deltaActivationGate, deltaOuputGate, deltaOfCellState, deltaOfHiddenState = self.LSTM.backwardPass(errorForLSTM, self.cellStateArray[time-1], self.forgetGateArray[time], self.inputGateArray[time], self.activationGateArray[time], self.outputGateArray[time], deltaOfCellState, deltaOfHiddenState)

            #update total delta of forget gate
            totalDeltaForgetGate += deltaForgetGate
            #update total delta of input gate
            totalDeltaInputGate += deltaInputGate
            #update total delta of activation gate
            totalDeltaActivationGate += deltaActivationGate
            #update total delta of output gate
            totalDeltaOutputGate += deltaOuputGate

        #update LSTM gates matrices with average value over batch
        self.LSTM.updateWeightMatrix(totalDeltaForgetGate/self.batchSize, totalDeltaInputGate/self.batchSize, totalDeltaActivationGate/self.batchSize, totalDeltaOutputGate/self.batchSize)
        #update RNN weight matrix with avergae value over batch
        self.updateWeightMatrix(deltaWeightUpdate/self.batchSize)

        return batchError

    #function to update weights of dense layer
    def updateWeightMatrix(self, avgDelta):
        self.updateDenseWeigthMatrix = self.gradientMulFactor * self.updateDenseWeigthMatrix + self.deltaMulFactor  * np.power(avgDelta, 2)
        gradientVal = self.updateDenseWeigthMatrix + self.denseWeightAddition
        self.denseWeigthMatrix -= self.learningRate/np.sqrt(gradientVal) * avgDelta

    #function to predcit output for test dataset
    def test(self, batchInputData, batchOutputData):

        #batch size of test dataset
        batchSize = batchInputData.shape[0]
        #actual output value
        self.acualOutput = batchOutputData
        #an array to store cell states after each data point
        self.cellStateArray = np.zeros((batchSize+1, self.hlSize))
        #an array to store final predicted value after each data point
        self.finalPredictedArray = np.zeros((batchSize+1, self.outputDimension))
        #an array to store lstm output after each data point
        self.hiddenStateArray = np.zeros((batchSize+1, self.hlSize))
        #an array to store forget gate values after each data point
        self.forgetGateArray = np.zeros((batchSize+1, self.hlSize))
        #an array to store input gate values after each data point
        self.inputGateArray = np.zeros((batchSize+1, self.hlSize))
        #an array to store activation gate values after each data point
        self.activationGateArray = np.zeros((batchSize+1, self.hlSize))
        #an array to store output gate values after each data point
        self.outputGateArray = np.zeros((batchSize+1, self.hlSize))

        #total test data error
        totalTestDataError = 0

        for time in range(1, batchSize):

            #append previous lstm output and present input arrays to give input to lstm cell
            self.LSTM.inputData = np.hstack((self.hiddenStateArray[time-1], batchInputData[time]))

            cellStateValue, hiddenStateValue, forgetGateValue, inputGateValue, activationGateValue, outputGateValue = self.LSTM.forwardPass()

            #store the above values into the arrays defined above in RNN
            self.cellStateArray[time] = cellStateValue
            self.hiddenStateArray[time] = hiddenStateValue
            self.forgetGateArray[time] = forgetGateValue
            self.inputGateArray[time] = inputGateValue
            self.activationGateArray[time] = activationGateValue
            self.outputGateArray[time] = outputGateValue
            #multiply the hidden state value with weight matrix and pass it through sigmoid to obtain predicted value
            self.finalPredictedArray[time] = self.sig(np.dot(self.denseWeigthMatrix, hiddenStateValue))

            errorForDataPoint = abs(self.finalPredictedArray[time] - batchOutputData[time])
            totalTestDataError += errorForDataPoint

        return self.finalPredictedArray, totalTestDataError
