import matplotlib.pyplot as mp
import numpy as np

def plotError():
    data = []
    with open("errorData.txt", "r") as file:
        for eachLine in file:
            if (eachLine != "0.0\n"):
                data.append(float(eachLine))

    lst = list(np.arange(1,len(data)+1))
    mp.plot(lst, data)
    mp.xlabel("Epochs")
    mp.ylabel("RMSE")
    mp.title("Error Over Time")
    mp.show()


plotError()