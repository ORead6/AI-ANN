import numpy as np
import random
import matplotlib.pyplot as mp

#Change to get dataset in format as shown
testData = np.matrix([[10.4,4.393,9.291,0,0,0,4], 
            [9.95,4.239,8.622,0,0,0.8,0],
            [9.46,4.124,8.057,0,0,0.8,0],
            [9.41,4.363,7.925,2.4,24.8,0.8,61.6],
            [26.3,11.962,58.704,11.2,5.6,33.6,111.2],
            [32.1,10.237,34.416,0,0,1.6,0.8]])

#Neurons is how long the test Data array is
I_dim = 7

#Hidden Neurons
H_dim = 2

#Output Neurons
O_dim = 1

#Learning Rate
learning_param = 0.1

def normaliseData(data, case):
#Assign realistic min and max values with the ranges of data
    maxVal = 100 #Look through data set and change this value appropriately
    minVal = 0

    #Normalise data set between 0-1 for compairson
    if (case == "pre"):
        for x in range(0, len(data)):
            data[x] = (data[x] - minVal) / (maxVal - minVal)

        data = np.matrix(data).transpose()
        return data
    
    #Taking data between 0-1 and changing to put through correct range of values
    elif (case == "post"):
        for x in range(0, len(data)):
            data[x] = (data[x] * (maxVal - minVal)) + minVal
        data = np.matrix(data).transpose()
        return data

def weightInit(dict):
    for i in range(0, len(dict)):
        for x in range(0, len(dict[i]["weights"])):
            dict[i]["weights"][x] = random.uniform(-(1/np.sqrt(I_dim)), (1/np.sqrt(I_dim)))

    return dict

def activation(x):
    #Sigmoid
    result = 1 / (1 + np.exp(-x))
    return result

def feedForward(inputs, hiddenNeurons, outputNeurons):
    for i in range(0, len(inputs)):
        thisPass = inputs[i]

        for x in range(0, H_dim):
            weights = hiddenNeurons[x]["weights"]
            wS = np.matmul((np.matrix(weights)), (np.matrix(thisPass)).transpose()) + hiddenNeurons[x]["bias"]
            hiddenNeurons[x]["act"] = activation(wS.item(0))

        for x in range(0, O_dim):
            weights = outputNeurons[x]["weights"]
            actVals = [neuron["act"] for neuron in hiddenNeurons]
            wS = np.matmul((np.matrix(weights)).transpose(), (np.matrix(actVals))) + outputNeurons[x]["bias"]
            outputNeurons[x]["val"] = activation(wS.item(0))


def main(data):
    #Feed Forward Algorithm
    feedForward(data, hidden, output)


#HiddenNeurons
hidden = []
for i in range(H_dim):
    hidden.append({"weights": ([1] * I_dim), "bias": 0, "act": 0}) 
#hidden = weightInit(hidden)

#Hidden -> Output Weights Neurons
output = []
for i in range(O_dim):
    output.append({"weights": ([2] * H_dim), "bias": 0}) 
#output = weightInit(output)


main(testData)