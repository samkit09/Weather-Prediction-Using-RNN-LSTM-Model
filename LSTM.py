# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:00:51 2019

@authors: Sahithi
          Hanumantha Rao
          Aishwarya
          Siva
"""
import numpy as np

class LSTM:
    #function to Initialize the LSTM cell with parameters
    def __init__ (self, inputDimension, outputDimension, batchSize, learningRate):

        #input data array
        self.inputData = np.zeros(inputDimension + outputDimension)
        #output data array
        self.outputData = np.zeros(shape = (1, outputDimension))
        #batch size
        self.batchSize = batchSize
        #an array to store cell state for a data point
        self.cellState = np.zeros(outputDimension)
        #learning rate
        self.learningRate = learningRate
        #an matrix to store input gate weights
        self.inputGateWeightMatrix = np.random.random((outputDimension, inputDimension + outputDimension))
        #an matrix to store forget gate weights
        self.forgetGateWeightMatrix = np.random.random((outputDimension, inputDimension + outputDimension))
        #an matrix to store output gate weights
        self.outputGateWeightMatrix = np.random.random((outputDimension, inputDimension + outputDimension))
        #an matrix to store activation gate weights
        self.activationGateWeightMatrix = np.random.random((outputDimension, inputDimension + outputDimension))
        #upate forget gate weight matrix
        self.updateForgetWeightMatrix = np.zeros_like(self.forgetGateWeightMatrix)
        #upate input gate weight matrix
        self.updateInputWeightMatrix = np.zeros_like(self.inputGateWeightMatrix)
        #upate activation gate weight matrix
        self.updateActivationWeightMatrix = np.zeros_like(self.activationGateWeightMatrix)
        #upate output gate weight matrix
        self.updateOutputWeightMatrix = np.zeros_like(self.outputGateWeightMatrix)

    #calculating sigmoid value
    def sigmoid(self, data):
        expVal = np.exp(data)
        sigmoidVal = expVal / (1 + expVal)
        return sigmoidVal
    
    #calculating the sigmoid derivative
    def sigmoidDerivative(self, data):
        sigmoidVal = self.sigmoid(data)
        derVal = sigmoidVal * (1 - sigmoidVal)
        return derVal
    
    #calculating derivative of hyperbolic tangent
    def tanhDerivative(self, data):
        val = np.tanh(data)
        return 1 - val * val

    #function for forward pass to calculate the cellstate and hidden state after each datapoint
    def forwardPass(self):

        #calculate gate values - activationgate uses tanh activation and all other gates use sigmoid activation
        inputGateValue = self.sigmoid(np.dot(self.inputGateWeightMatrix, self.inputData))
        forgetGateValue = self.sigmoid(np.dot(self.forgetGateWeightMatrix, self.inputData))
        outputGateValue = self.sigmoid(np.dot(self.outputGateWeightMatrix, self.inputData))
        activationGateValue = np.tanh(np.dot(self.activationGateWeightMatrix, self.inputData))

        #calculating cell state
        self.cellState = self.cellState*forgetGateValue + (activationGateValue*inputGateValue)

        #calculating output value(hidden state)
        self.outputData = outputGateValue * np.tanh(self.cellState)

        return self.cellState, self.outputData, forgetGateValue, inputGateValue, activationGateValue, outputGateValue

    #function for back propogation of error from RNN and to calculate the gate weight updates
    def backwardPass(self, errorFromRNN, previousCellState, forgetGateValue, inputGateValue, activationGateValue, outputGateValue, deltaOfCellState, deltaOfHiddenState):

        #sum error from RNN and hidden state
        errorFromRNN = errorFromRNN + deltaOfHiddenState
        #calculating derivatives of each gate
        derivativeOfOutputGate = np.tanh(self.cellState) * errorFromRNN
        derivativeCellState = errorFromRNN * outputGateValue * self.tanhDerivative(self.cellState) + deltaOfCellState
        deltaPreviousCellState = derivativeCellState * forgetGateValue
        derivativeOfForgetGate = derivativeCellState * previousCellState
        derivativeOfInputGate = derivativeCellState * activationGateValue
        derivativeOfActivationGate = derivativeCellState * inputGateValue

        #calculating delta of each gate
        inputDatainAtleast2d = np.atleast_2d(self.inputData)
        outGate = np.atleast_2d(derivativeOfOutputGate * self.sigmoidDerivative(outputGateValue)).T
        deltaOutputGate = np.dot(outGate, inputDatainAtleast2d)
        forgetGate = np.atleast_2d(derivativeOfForgetGate * self.sigmoidDerivative(forgetGateValue)).T
        deltaForgetGate = np.dot(forgetGate, inputDatainAtleast2d)
        inGate = np.atleast_2d(derivativeOfInputGate * self.sigmoidDerivative(inputGateValue)).T
        deltaInputGate = np.dot(inGate , inputDatainAtleast2d)
        actGate = np.atleast_2d(derivativeOfActivationGate * self.tanhDerivative(activationGateValue)).T
        deltaActivationtGate = np.dot(actGate, inputDatainAtleast2d)
        #calculating previous hidden state delta
        deltaPreviousHiddenState = np.dot(derivativeOfActivationGate, self.activationGateWeightMatrix)[:self.outputData.shape[0]]+ np.dot(derivativeOfOutputGate, self.outputGateWeightMatrix)[:self.outputData.shape[0]]+ np.dot(derivativeOfInputGate, self.inputGateWeightMatrix)[:self.outputData.shape[0]]+ np.dot(derivativeOfForgetGate, self.forgetGateWeightMatrix)[:self.outputData.shape[0]]
        #sending the updated gradient for all the calculated values
        return deltaForgetGate, deltaInputGate, deltaActivationtGate, deltaOutputGate, deltaPreviousCellState, deltaPreviousHiddenState

    #function to update gate weights from the deltas calculated in backpropagation
    def updateWeightMatrix(self, deltaForgetGate, deltaInputGate, deltaActivationtGate, deltaOutputGate):
        #update gradients for all gates - used to reduce the vanishing gradient problem
        self.updateForgetWeightMatrix = 0.85 * self.updateForgetWeightMatrix + 0.15 * deltaForgetGate * deltaForgetGate
        self.updateInputWeightMatrix = 0.85 * self.updateInputWeightMatrix + 0.15 * deltaInputGate * deltaInputGate
        self.updateActivationWeightMatrix = 0.85 * self.updateActivationWeightMatrix + 0.15 * deltaActivationtGate * deltaActivationtGate
        self.updateOutputWeightMatrix = 0.85 * self.updateOutputWeightMatrix + 0.15 * deltaOutputGate * deltaOutputGate

        #update the weight matrices of each gate with updated gradient matrices
        self.forgetGateWeightMatrix -= self.learningRate/np.sqrt(np.power(10.0, -8) + self.updateForgetWeightMatrix) * deltaForgetGate
        self.inputGateWeightMatrix -= self.learningRate/np.sqrt(np.power(10.0, -8) + self.updateInputWeightMatrix) * deltaInputGate
        self.activationGateWeightMatrix -= self.learningRate/np.sqrt(np.power(10.0, -8) + self.updateActivationWeightMatrix) * deltaActivationtGate
        self.outputGateWeightMatrix -= self.learningRate/np.sqrt(np.power(10.0, -8) + self.updateOutputWeightMatrix) * deltaOutputGate
