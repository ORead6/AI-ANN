from opcode import hasjabs
import numpy as np
import random
import matplotlib.pyplot as mp

#Change to get dataset in format as shown
testData = [[10.4,4.393,9.291,0,0,0,4], 
            [9.95,4.239,8.622,0,0,0.8,0],
            [9.46,4.124,8.057,0,0,0.8,0],
            [9.41,4.363,7.925,2.4,24.8,0.8,61.6],
            [26.3,11.962,58.704,11.2,5.6,33.6,111.2],
            [32.1,10.237,34.416,0,0,1.6,0.8]]

desiredOut =  [26.1, 24.86, 23.6, 23.47, 60.7, 98.01]

#Neurons is how long the test Data array is
I_dim = 7

#Hidden Neurons
H_dim = 1

#Output Neurons
O_dim = 1

#Learning Rate
learning_param = 0.1

#Amount of epochs
epochCount = 5

def normaliseData(data, case):
#Assign realistic min and max values with the ranges of data
    maxVal = 100 #Look through data set and change this value appropriately
    minVal = 0

    #Normalise data set between 0-1 for compairson
    if (case == "pre"):
        data = (data - minVal) / (maxVal - minVal)
        return data
    
    #Taking data between 0-1 and changing to put through correct range of values
    elif (case == "post"):
        data = (data * (maxVal - minVal)) + minVal
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

def derivative(x):
    return (x - (1 - x))

def plotGraph(errorGraph):
    print(errorGraph)
    mp.plot([x for x in range(len(errorGraph))], errorGraph)
    mp.xlabel("Epochs")
    mp.ylabel("Error")
    mp.title("Error Over Time")
    mp.show()

def backProp(dataIndex, hidNeurons, outNeurons, observed):
    desiredOutVal = observed[dataIndex]
    desiredOutVal = normaliseData(desiredOutVal, "pre")
    inpVal = testData[dataIndex]
    overallError = []

    for i in range(0, len(outNeurons)):
        error = abs(outNeurons[i]["val"] - desiredOutVal)
        overallError.append(error)
        outNeurons[i]["delta"] = (desiredOutVal - outNeurons[i]["val"]) * (derivative(outNeurons[i]["val"]))


    for i in range(0, len(hidNeurons)):
        for j in range(0, len(outNeurons)):
            weightToOut = outNeurons[j]["weights"][i] 
            outDelta = outNeurons[j]["delta"]
            nodeVal = hidNeurons[i]["act"]
            hidNeurons[i]["delta"] = (weightToOut * outDelta * derivative(nodeVal))


def getOverall(data):
    sum = 0
    for i in range(0, len(data)):
        sum += data[i]
    
    return (sum / len(data))

def wSum(m1, m2):
    sum = 0
    for i in range(0, len(m1)):
        sum += m1[i]*m2[i]
        

    return sum

def feedForward(inputs, hiddenNeurons, outputNeurons, observedVal):
    epochErrors = []
    for j in range(epochCount):
        errors = []
        for i in range(0, len(inputs)):
            thisPass = inputs[i]

            for x in range(0, H_dim):
                weights = hiddenNeurons[x]["weights"]
                wS = wSum(weights, thisPass) + hiddenNeurons[x]["bias"]
                hiddenNeurons[x]["act"] = activation(wS)

            for x in range(0, O_dim):
                weights = outputNeurons[x]["weights"]
                actVals = [neuron["act"] for neuron in hiddenNeurons]
                #Add bias
                wS = wSum(weights, actVals)
                outputNeurons[x]["val"] = activation(wS)

                error = abs(outputNeurons[x]["val"] - normaliseData(observedVal[i], "pre"))
                errors.append(error)

            backProp(i, hiddenNeurons, outputNeurons, observedVal)

        epochErrors.append(getOverall(errors))

    return(epochErrors)

            
        

def main(data, endData):
    #Feed Forward Algorithm
    plotGraph(feedForward(data, hidden, output, endData))
    


#HiddenNeurons
hidden = []
for i in range(H_dim):
    hidden.append({"weights": ([1] * I_dim), "bias": 0, "act": 0}) 
hidden = weightInit(hidden)

#Hidden -> Output Weights Neurons
output = []
for i in range(O_dim):
    output.append({"weights": ([2] * H_dim), "bias": 0}) 
output = weightInit(output)


main(testData, desiredOut)
