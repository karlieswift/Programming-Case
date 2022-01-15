import csv
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns

class LinearRegression:
    def __init__(self):
        '''
        设置画布，显示中文
        '''
        sns.set()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    def loadDataSet(self,file):
        '''
        :param file: 加载数据
        :return: 返回数据X,y
        '''
        data = []
        label = []
        i = 0
        for row in csv.reader(open(file,encoding='utf-8-sig')):
            if i > 0:
                data.append([1.0, float(row[0])])
                label.append(float(row[1]))
            i = i + 1
        return data, label

    def analytical_solution(self,data,label):
        '''
        解析解直接通过公式计算，不需要迭代
        :param data: X
        :param label: y
        :return: 线性回归的系数
        '''
        data=mat(data)
        label=mat(label).T
        theta = linalg.inv(data.transpose() * data) * data.transpose() * label
        return theta

    def gradientDescent(self,data,label):
        '''
        梯度下降法
        :param data: 数据
        :param label:
        :return: 学习系数
        '''
        dataMat = mat(data)
        labelMat = mat(label).transpose()
        n,m = dataMat.shape
        theta = ones((m,1))
        error = dataMat * theta - labelMat
        precost = 1 / 2 * error.transpose() * error
        # 画图部分
        plt.ion()
        xs = [0, 0]
        ys= [0, precost[0, 0]]
        # 画图部分
        for k in range(10000):
            theta = theta - 0.0001 * (dataMat.transpose() * error)
            error = dataMat * theta - labelMat
            cost = 1/2 * error.transpose() * error

            xs[0] = xs[1]
            ys[0] = ys[1]
            xs[1] = k
            ys[1] = cost[0, 0]
            plt.figure(1)
            plt.title('损失函数', fontsize=14)
            plt.xlabel('迭代次数', fontsize=8)
            plt.ylabel('损失', fontsize=8)
            plt.plot(xs, ys, color = 'red')
            plt.figure(2)
            self.plot_lr(data, label, theta, '梯度下降法',2014)
            plt.pause(0.1)
            if abs(precost - cost) <  0.0002:   # cost变化已不大，收敛
                break
            precost = cost
        return theta

    def plot_lr(self,data, label, theta, title,year):
        '''
        将学习过程画图
        :param data: X特征
        :param label: y目标
        :param theta:
        :param title: 标题
        :param year: 打印年份
        :return:
        '''

        def plot_year(year):
            i=year-2000
            plt.plot(i, y[i][0], 'om')
            plt.text(i, y[i][0] + 0.8, '{}年'.format(year), color='orange', fontsize=10)
            plt.text(i, y[i][0], '%.2f' % y[i][0], color='orange', fontsize=10)


        plt.clf()
        x = arange(0, 20).tolist()
        y = (theta[0] + theta[1] * x).T.tolist()
        ax = plt.subplot()
        plt.title(title , fontsize=14)
        plt.xlabel('年份', fontsize=8)
        plt.xticks(x, list((i+2000 for i in range(20))))
        plt.xticks(rotation=66)
        plt.ylabel('价格', fontsize=8)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        plt.xlim(0, 16)
        plt.ylim(0, 16)
        plt.plot([data[i][1] for i in range(0, 14)], label, "ob")
        plot_year(year=year)
        plt.plot(x, y, color='red')

if __name__=='__main__':
    lr=LinearRegression()
    data, label = lr.loadDataSet("data.csv")
    theta = lr.gradientDescent(data, label)
    print("梯度下降法系数:",theta)
    plt.ioff()
    plt.figure(3)
    theta = lr.analytical_solution(data, label)
    lr.plot_lr(data, label, theta, '解析解法',2014)
    print("解析解系数：",theta)
    plt.show()



