import matplotlib.pyplot as mp
import numpy as np

def plotError():
    data = []
    with open("errorData.txt", "r") as file:
        for eachLine in file:
            data.append(float(eachLine))

    lst = list(np.arange(1,len(data)+1))
    mp.plot(lst, data)
    mp.xlabel("Epochs")
    mp.ylabel("RMSE")
    mp.title("Error Over Time")
    mp.show()

def plotDot():
    dataOne = []
    dataTwo = []
    with open("dotGraph.txt", "r") as file:
        for eachLine in file:
            temp = eachLine.split(",")
            dataOne.append(temp[0][0:-1])
            dataTwo.append(temp[1][0:-1])

    mp.plot(dataOne, dataOne, 'r--', dataOne, dataTwo, 'bs')
    mp.xlabel("Desired Output")
    mp.ylabel("Predicted Output")
    mp.title("Error Over Time")

    mp.gca()
    mp.show()

def main():
    userInp = str(input("Dot or Error or Stop\n"))

    while userInp != "stop":
        if userInp == "dot":
            plotDot()
        elif userInp == "Error":
            plotError()

        userInp = str(input("Dot or Error or Stop"))

main()