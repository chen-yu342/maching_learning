import numpy as np
from  numpy import *

def loadDataSet():
    dataMat = [];labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatix = mat(dataMatIn) #转为矩阵
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatix)
    alpha = 0.001
    max



if __name__ == '__main__':
    mm = array((1,2,3))
    nn = array((1,2,3))
    print(mm+nn)
    #loadDataSet()
