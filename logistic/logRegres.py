import numpy as np
from  numpy import *
import matplotlib.pyplot as plt

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
    #将类别标签classLabels转换为矩阵形式，并使用.transpose()方法将其转置。转置后的矩阵存储在变量labelMat中。这一步是为了为接下来的矩阵乘法做准备。
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatix)
    #定义学习率alpha并赋值为0.001。这是梯度上升算法中用于更新权重的重要参数。
    alpha = 0.001
    maxCycles = 500
    #初始化权重矩阵weights，其大小为(n, 1)，并且所有元素初始值都为1。这是因为在逻辑回归中，权重通常初始化为正数。
    weights = ones((n,1))
    for k in range(maxCycles):
        #计算当前的预测值h。这是通过将数据矩阵dataMatix与权重矩阵weights相乘，然后应用sigmoid函数得到的。sigmoid函数将结果压缩到0和1之间的范围，表示概率。
        h = sigmoid(dataMatix * weights)
        #计算误差矩阵error，它是真实标签labelMat与预测值h之间的差异。这个误差用于梯度上升算法中更新权重。
        error = (labelMat -h)
        #根据梯度上升算法，更新权重矩阵weights。更新的方法是：当前权重加上学习率乘以误差乘以数据矩阵的转置。这个公式是梯度上升算法的核心，它根据误差来调整权重。
        weights = weights +alpha * dataMatix.transpose()*error
    #最后，函数返回更新后的权重矩阵。
    return weights

#这个函数的目标是通过绘制散点图和分类边界来可视化线性分类器的结果。
def plotBestFit(W):
    # 把训练集数据用坐标的形式画出来
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    #这里，首先将数据矩阵转换为NumPy数组，然后获取数组的形状（行数和列数）。
    #接着，创建四个空列表以存储两类数据的x和y坐标。
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    #这个循环遍历每一个数据点，并根据其标签将数据点分类。如果是类别1，
    #数据点被添加到 xcord1 和 ycord1；如果是类别0，数据点被添加到 xcord2 和 ycord2。
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # 把分类边界画出来
    x = np.arange(-3.0,3.0,0.1)
    #这部分代码首先定义了一个x值的范围，然后计算对应的y值。这里的计算方式是基于线性分类器的公式 y = (W0 + W1*x) / W2，
    #其中W0、W1和W2是学习得到的权重参数。最后，使用plot函数把分类边界画出来。
    y = (-W[0]-W[1]*x)/W[2]
    ax.plot(x,y)
    plt.show()


if __name__ == '__main__':
    dataMat,labelMat = loadDataSet()
    weights = gradAscent(dataMat,labelMat)
    print(weights)
    print('==============='*2)
    print(weights.getA())
    plotBestFit(weights.getA())

