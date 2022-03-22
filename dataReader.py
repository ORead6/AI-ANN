import matplotlib.pyplot as mp
import numpy as np

data = []
with open("errorData.txt", "r") as file:
    for eachLine in file:
        data.append(float(eachLine))

def plotGraph(errorGraph):
    print(errorGraph[-1])
    lst = list(np.arange(1,len(errorGraph)+1))
    mp.plot(lst, errorGraph)
    mp.xlabel("Epochs")
    mp.ylabel("RMSE")
    mp.title("Error Over Time")
    mp.show()

plotGraph(data)