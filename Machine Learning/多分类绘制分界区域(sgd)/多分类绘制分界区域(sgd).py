
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# 加载数据集，最后一列最为类别标签，前面的为特征属性的值
def loadDataSet(x,y):
    # 生成X和y矩阵
    #dataMat = np.mat(x)
    y = np.mat(y)
    b = np.ones(y.shape)  # 添加全1列向量代表b偏量
    X = np.column_stack((b, x))  # 特征属性集和b偏量组成x
    X = np.mat(X)
    labeltype = np.unique(y.tolist())       # 获取分类数目
    eyes = np.eye(len(labeltype))    # 每一类用单位矩阵中对应的行代替，表示目标概率。如分类0的概率[1,0,0]，分类1的概率[0,1,0]，分类2的概率[0,0,1]
    Y=np.zeros((X.shape[0],len(labeltype)))
    for i in range(X.shape[0]):
        Y[i,:] = eyes[int(y[i,0])]               # 读取分类，替换成概率向量。这就要求分类为0,1,2,3,4,5这样的整数
    return X, y,Y       #X为特征数据集，y为分类数据集，Y为概率集

# 数据处理，x增加默认值为1的b偏量，y处理为onehot编码类型
def data_convert(x,y):
    print(y.shape)
    b=np.ones(y.shape)   # 添加全1列向量代表b偏量
    # b=np.ones((120,1))   # 添加全1列向量代表b偏量
    x_b=np.column_stack((b,x)) # b与x矩阵拼接
    K=len(np.unique(y.tolist())) # 判断y中有几个分类
    eyes_mat=np.eye(K)           # 按分类数生成对角线为1的单位阵
    y_onehot=np.zeros((y.shape[0],K)) # 初始化y的onehot编码矩阵
    for i in range(0,y.shape[0]):
        y_onehot[i]=eyes_mat[y[i]]  # 根据每行y值，更新onehot编码矩阵
    return x_b,y,y_onehot

# softmax函数，将线性回归值转化为概率的激活函数。输入s要是行向量
def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=1)

# 逻辑回归中使用梯度下降法求回归系数。逻辑回归和线性回归中原理相同，只不过逻辑回归使用sigmoid作为迭代进化函数。
def gradAscent(x, y,alpha=0.05,max_loop=500):
    # 梯度上升算法
    #alpha=0.05  # 步长
    #max_loop=500 # 循环次数

    weights = np.ones((x.shape[1],y.shape[1]))             #初始化权回归系数矩阵  系数矩阵的行数为特征矩阵的列数，系数矩阵的列数为分类数目
    print('weights初始化值：',weights)
    for k in range(max_loop):
        # k=0
        h =  softmax(x*weights)                                #梯度上升矢量化公式，计算预测值（行向量）。每一个样本产生一个概率行向量
        error = h-y                                            #计算每一个样本预测值误差
        weights = weights - alpha * x.T * error                   # 根据所有的样本产生的误差调整回归系数
        #print('k:',k,'weights:',weights)
    return weights                                                     # 将矩阵转换为数组，返回回归系数数组

def SoftmaxGD(x,y,alpha=0.05,max_loop=500):
    # 梯度上升算法
#    alpha=0.05  # 步长
#    max_loop=500 # 循环次数
    #x = StandardScaler().fit_transform(x) # 数据进行标准化
    m=np.shape(x)[1]  # x的特征数
    n=np.shape(y)[1]  # y的分类数
    weights=np.ones((m,n)) # 权重矩阵

    for k in range(max_loop):
        # k=2
        h=softmax(x*weights)
        error=y-h
        weights=weights+alpha*x.transpose()*error # 梯度下降算法公式
    #print('k:',k,'weights:',weights.T)
    return weights.getA()

def SoftmaxSGD(x,y,alpha=0.05,max_loop=50):
    # 随机梯度上升算法
#    alpha=0.05
#    max_loop=500
    #x = StandardScaler().fit_transform(x) # 数据进行标准化

    m=np.shape(x)[1]
    n=np.shape(y)[1]
    weights=np.ones((m,n))

    for k in range(max_loop):
        for i  in range(0,len(x)):
            # k=0;i=0
            h=softmax(x[i]*weights)
            error=y[i]-h[0]
            weights=weights+alpha*x[i].T*error[0]  # 随机梯度下降算法公式
            #print('k:',k,'i:',i,'weights:',weights.T)
    return weights.getA()

# 多分类只能绘制分界区域。而不是通过分割线来可视化
def plotBestFit(dataMat,labelMat,weights):

    # 获取数据边界值，也就属性的取值范围。
    x1_min, x1_max = dataMat[:, 1].min() - .5, dataMat[:, 1].max() + .5
    x2_min, x2_max = dataMat[:, 2].min() - .5, dataMat[:, 2].max() + .5
    # 产生x1和x2取值范围上的网格点，并预测每个网格点上的值。
    step = 0.02
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    testMat = np.c_[xx1.ravel(), xx2.ravel()]   #形成测试特征数据集
    testMat = np.column_stack((np.ones(((testMat.shape[0]),1)),testMat))  #添加第一列为全1代表b偏量
    testMat = np.mat(testMat)
    # 预测网格点上的值
    y = softmax(testMat*weights)   #输出每个样本属于每个分类的概率
    # 判断所属的分类
    predicted = y.argmax(axis=1)                            #获取每行最大值的位置，位置索引就是分类
    predicted = predicted.reshape(xx1.shape).getA()
    # 绘制区域网格图
    plt.pcolormesh(xx1, xx2, predicted, cmap=plt.cm.Paired)

    # 再绘制一遍样本点，方便对比查看
    plt.scatter(dataMat[:, 1].flatten().A[0], dataMat[:, 2].flatten().A[0],
                c=labelMat.flatten().A[0],alpha=.5)  # 第一个偏量为b，第2个偏量x1，第3个偏量x2
    plt.show()

# 对新对象进行预测
def predict(weights,testdata):
    y_hat=softmax(testdata*weights)
    predicted=y_hat.argmax(axis=1).getA()
    return predicted

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    standardScaler=StandardScaler()
    #3分类
    x=pd.read_table('./Iris/train/x.txt',sep=' ')
    y= pd.read_table('./Iris/train/y.txt')
    # 2分类
    # x=pd.read_table('./Exam/train/x.txt',sep=' ')
    # y= pd.read_table('./Exam/train/y.txt')
    x=standardScaler.fit_transform(x)

    x=np.array(x)

    y=np.array(y.iloc[:,0])

    y=np.array(y)



    plt.scatter(x[:,0],x[:,1],c=y,s=8) # 画图查看数据图像

    # 转换数据为matrix类型
    x=np.mat(x)
    y=np.mat(y).T

    # 调用数据预处理函数
    X,Y,Y_onehot=data_convert(x,y)

    # 批量梯度下降算法
    weights1=SoftmaxGD(X, Y_onehot)
    print('批量梯度下降算法')
    print(weights1)
    plotBestFit(X,Y,weights1)
    y_hat1=predict(weights1,X)
    print()

    # 随机批量梯度下降算法
    weights2=SoftmaxSGD(X, Y_onehot)
    print('随机批量梯度下降算法')
    print(weights2)
    plotBestFit(X,Y,weights2)
    y_hat2=predict(weights2,X)