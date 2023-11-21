# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:09:13 2019

@authors: Sahithi
		  Hanumantha Rao
		  Aishwarya
		  Siva
"""
import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
from io import BytesIO
from zipfile import ZipFile
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from RNN import RecurrentNeuralNetwork

#function to append next row global active power value to current row
def appendRowAbove(inputData):
    rowCount = inputData.shape[0]
    columnCount = inputData.shape[1]    
    returnData = np.zeros(shape = (rowCount-1, columnCount+1))      
    for i in range(rowCount-1):
        for j in range(columnCount+1):
            if(j < columnCount):
                returnData[i][j] = inputData[i][j]
            else:
                returnData[i][j] = inputData[i+1][0]
    finalData = pd.DataFrame(returnData)    
    columnNames = [('golbal_active_power'), ('golbal_reactive_power'), ('voltage'), ('golbal_intensity'), ('sub_metering_1'), ('sub_metering_2'), ('sub_metering_3')
                    ,('golbal_active_power_next_row')]
    finalData.columns = columnNames
    return finalData

#function to download data file from internet
def getFileFromWeb(fileURL):
    webURL = requests.get(fileURL)
    zippedFile = ZipFile(BytesIO(webURL.content))
    zippedNames = zippedFile.namelist()    
    fileName = zippedNames.pop()
    unzippedFile = zippedFile.open(fileName)
    return unzippedFile    

#read data from csv file using pandas read_csv function
energyData = pd.read_csv(getFileFromWeb("http://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"), sep = ';', 
                         parse_dates = {'dateTime' : ['Date', 'Time']}, infer_datetime_format = True, 
                         low_memory = False, na_values = ['nan', '?'], index_col = 'dateTime')

#if there were missing values in the data fill them with mean value of that column
for column in range(0, energyData.shape[1]):
    energyData.iloc[:, column] = energyData.iloc[:, column].fillna(energyData.iloc[:, column].mean())

#as the data is large, aggregating them to hourly data from per minute data
aggData = energyData.resample('h').mean()
#get the data in matrix format
matrixData = aggData.values 
#transform the data of feature into range between 0 and 1
transformFunc = MinMaxScaler(feature_range = (0, 1))
transformedData = transformFunc.fit_transform(matrixData) 
#final dataset after appending next row to previous row
finalData = appendRowAbove(transformedData)


#now we have our data, lets divide them into train and test datasets
matrixFormData = finalData.values
#number of data instances in the dataset
rowCount = finalData.shape[0]
#percentage split of dataset into train and test datasets
train_test_split_percentage = 0.8
#number of data instances in the training data
train_rows = round(rowCount*train_test_split_percentage)
#training data with input features and output data
train_X, train_y = matrixFormData[:train_rows, :-1], matrixFormData[:train_rows, -1]
#test data with input features and output data
test_X, test_y = matrixFormData[train_rows:, :-1], matrixFormData[train_rows:, -1]

print("Starting traning of the dataset.............")
#initializing RNN with required parameters
learningRateList = [0.00001, 0.0001, 0.001, 0.005, 0.1]
batchSizeList = [2, 5, 10, 30, 50, 100]
hiddenLayerSizeList = [2, 5, 10, 20, 50, 100]
epochList = [2, 5, 10, 15, 20, 50, 100]

#a list is created for each of the parameters and outputs
epochsList = []
hiddenLayerSizesList = []
learningRatesList = []
batchSizesList = []
RMSEListForTrain = []
RMSEListForTest = []

#for epoch, hiddenLayerSize, learningRate, batchSize in zip(epochList, hiddenLayerSizeList, learningRateList, batchSizeList):    

for epoch in epochList:    
    for hiddenLayerSize in hiddenLayerSizeList:
        for learningRate in learningRateList:
            for batchSize in batchSizeList: 
                RNN = RecurrentNeuralNetwork(train_X, train_y, hiddenLayerSize, learningRate, batchSize)
                for epochNumber in range(0, epoch):
                    rowStart = 0 
                    totalTrainError = 0
                    mseBatchError = 0
                                    
                    for rowEnd in range(batchSize, train_X.shape[0], batchSize):
                        #RNN forwward pass
                        predictedValue = RNN.forwardPass(train_X[rowStart : rowEnd])
                        #update all our weights using our error
                        totalTrainError  += RNN.backwardPass(train_y[rowStart : rowEnd], train_X[rowStart : rowEnd])
                        #mse value                                                                                                
                        predictedValue = np.concatenate((predictedValue[1:], train_X[rowStart : rowEnd, -7 : -1]), axis = 1)
                        predictedValue = transformFunc.inverse_transform(predictedValue)
                        predictedValue = predictedValue[:, 0]                                                
                        trainY = np.concatenate((train_y[rowStart : rowEnd].reshape(train_y[rowStart : rowEnd].shape[0], 1), train_X[rowStart : rowEnd, -7 : -1]), axis = 1)
                        trainY = transformFunc.inverse_transform(trainY)
                        trainY = trainY[:, 0]
                        mseBatchError += mean_squared_error(trainY, predictedValue)               
                        rowStart = rowEnd                        
                    
                    #root mean square error for train data
                    trainRMSE = np.sqrt(mseBatchError*batchSize/train_rows)
                
                #append variables to their lists to use for output tables
                epochsList.append(epoch)                
                hiddenLayerSizesList.append(hiddenLayerSize)
                learningRatesList.append(learningRate)
                batchSizesList.append(batchSize)
                RMSEListForTrain.append(trainRMSE)                
                
                #call test function to predict values for test dataset as the training is done and model is ready
                predictedValues, testError = RNN.test(test_X, test_y)
                predictedValues = np.concatenate((predictedValues[1:], test_X[:, -7 : -1]), axis = 1)
                predictedValues = transformFunc.inverse_transform(predictedValues)
                predictedValues = predictedValues[:, 0]                                                
                testY = np.concatenate((test_y.reshape(test_y.shape[0], 1), test_X[ : , -7 : -1]), axis = 1)
                testY = transformFunc.inverse_transform(testY)
                testY = testY[:, 0]
                #calculate mean squared error using in-built function
                mseTestError = mean_squared_error(testY, predictedValues)
                #root mean squared error for test data
                testRMSE = np.sqrt(mseTestError)
                #append to the respective list
                RMSEListForTest.append(testRMSE)
                print("epochs : " , epoch , ", hiddenLayerSize : " , hiddenLayerSize , ", learningRate : " , learningRate , ", batchSize : " , batchSize , ", TrainRMSE : " , trainRMSE , ", TestRMSE : " , testRMSE)                


dataFrame = DataFrame({'epochs' : epochsList, 'hiddenLayerSize' : hiddenLayerSizesList, 'learningRate' : learningRatesList, 'batchSize' : batchSizesList, 'RMSE for Train' : RMSEListForTrain, 'RMSE for Test' : RMSEListForTest})
dataFrame.to_excel('output.xlsx', sheet_name = 'output', index = False)
