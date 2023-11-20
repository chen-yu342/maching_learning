import numpy as np
from  numpy import *


def loadDataset():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] =1
        else:
            print(f'the word {word} is not in my Vocabulary!')
    return returnVec

#生成训练集向量列表 trainmat 所有词条向量组成的表
def get_trainMst(dataSet):
    trainmat = [] #初始化向量列表
    vocabList = createVocabList(dataSet) #生成词汇表
    for inputSet in dataSet:
        returnVec = setOfWords2Vec(vocabList,inputSet) #将当前的词条向量化
        trainmat.append(returnVec)
    return trainmat

def trainNB(trainMat,classVec):
    n = len(trainMat) #计算训练的文档数目
    m = len(trainMat[0]) #计算每篇文档的词条数
    pAb = sum(classVec)/n #文档属于侮辱类的概率
    p0Num = np.ones(m) #词条出现初始化为1
    p1Num = np.ones(m) #词条出现初始化为1
    p0Demo = 2 #分母初始化为2
    p1Demo = 2 #分母初始化为2
    for i in range(n):#遍历每一个文档
        if classVec[i] ==1: #统计属于侮辱类的条件概率所需的数据
            p1Num += trainMat[i]
            p1Demo += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Demo += sum(trainMat[i])

    p1v = np.log(p1Num/p1Demo)
    p0v = np.log(p0Num/p0Demo)
    return p0v,p1v,pAb

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB(testVec):
    listOposts,listClasses = loadDataset()
    myVocablist = createVocabList(listOposts)
    trainMat = get_trainMst(listOposts)
    p0v,p1v,pAb = trainNB(trainMat,listClasses)
    thisones = setOfWords2Vec(myVocablist,testVec)
    if classifyNB(thisones,p0v,p1v,pAb)==1:
        print(testVec,'属于侮辱类')
    else:
        print(testVec,'属于非侮辱类')


if __name__ == '__main__':
    # listOposts,listClasses = loadDataset()
    # myVocablist = createVocabList(listOposts)
    #print(myVocablist)
    #print(get_trainMst(myVocablist))
    # trainMat = []
    # for postinDoc in listOposts:
    #     trainMat.append(setOfWords2Vec(myVocablist,postinDoc))
    # #print(trainMat)
    # p0v,p1v,pAb = trainNB(trainMat,listClasses)
    # print(p0v)
    # print(p1v)
    # print(pAb)
    testVec1 = ['love','my','dalmation']
    testingNB(testVec1)
    testVec1 = ['stupid', 'garbage']
    testingNB(testVec1)



