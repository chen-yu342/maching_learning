import numpy as np
from  numpy import *
import re

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

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index[word]] += 1
    return returnVec

def textParse(bigSting):
    listOfTokens = re.split(r'\W+',bigSting)
    return [tok.lower() for  tok in listOfTokens if len(tok) >2]




def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):  # 遍历25个txt文件
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())  # 读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())  # 读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)  # 标记正常邮件，0表示正常文件
    vocabList = createVocabList(docList)  # 创建词汇表，不重复
    trainingSet = list(range(50))
    testSet = []  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索索引值
        testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
        del (trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值
    trainMat = []
    trainClasses = []  # 创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB(np.array(trainMat), np.array(trainClasses))  # 训练朴素贝叶斯模型
    errorCount = 0  # 错误分类计数
    for docIndex in testSet:  # 遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])  # 测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))

if __name__ == '__main__':
    spamTest()
    # mySent = 'This book is the best book on Python or M.L I have ever laid eyes upon.'
    # #print(mySent.split())
    # regEx = re.compile('\\W+')
    # listOfTokens = regEx.split(mySent)
    # #print(listOfTokens)
    # listOfTokens = [tok.lower() for tok in listOfTokens if len(tok) > 0]
    # print(listOfTokens)




