"""
 Env: /anaconda3/python3.7
 Time: 2021/10/29 14:05
 Author: karlieswfit
 File: LinearRegression.py
 Describe:梯度下降法需要进行标准化
 方法1：直接通过公式计算系数fit(self, x_train, y_train)
 方法2：通过梯度下降法求系数fit_gd(self, x_train, y_train, lr=0.001, max_iter=10000, epsilon=1e-4)
 方法3：通过随机梯度下降法求系数fit_sgd(self, x_train, y_train,lr=0.01,max_iter=1000)
 fit_gd_plot(self, x_train, y_train, lr=0.001, max_iter=10000, epsilon=1e-0): 将loss函数动态可视化（学习过程）


"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self.intercept = None  # 截距
        self.coef = None  # 系数
        self._theta = None

    # 方法1：直接通过公式计算系数
    def fit(self, x_train, y_train):
        # 初始化数据样本，将截距b的系数1加入到矩阵。
        X = np.hstack((np.ones(shape=(len(x_train), 1)), x_train))
        # 直接通过公式计算系数
        self._theta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y_train)

        self.intercept = self._theta[0]  # 截距
        self.coef = self._theta[1:]  # 系数

        # 方法2：通过梯度下降法求系数

    def fit_gd(self, x_train, y_train, lr=0.001, max_iter=10000, epsilon=1e-4):
        def loss(X, y, theta):
            return ((X.dot(theta) - y) ** 2).sum() / len(X)

        def clau_gradient(X, y, theta):
            return X.T.dot(X.dot(theta) - y) * 2 / len(x_train)

        X = np.hstack((np.ones(shape=(len(x_train), 1)), x_train))
        theta = np.zeros(X.shape[1])
        i = 0
        while i < max_iter:
            i = i + 1
            gradient = clau_gradient(X, y_train, theta)
            theta = theta - lr * gradient
            if loss(X, y_train, theta) < epsilon:
                break

        self._theta = theta
        self.intercept = self._theta[0]
        self.coef = self._theta[1:]

    def fit_gd_plot(self, x_train, y_train, lr=0.001, max_iter=10000, epsilon=1e-0):
        """
        将loss函数动态可视化（学习过程）
        """

        def loss(X, y, theta):
            return ((X.dot(theta) - y) ** 2).sum() / len(X)

        def clau_gradient(X, y, theta):
            return X.T.dot(X.dot(theta) - y) * 2 / len(x_train)

        X = np.hstack((np.ones(shape=(len(x_train), 1)), x_train))
        theta = np.zeros(X.shape[1])
        i = 0

        cost_1 = 0.5 * loss(X, y_train, theta).T * loss(X, y_train, theta)
        plt.ion()
        plt.figure(1)
        X_S = [0, 0]
        Y_S = [0, cost_1]
        while i < max_iter:
            X_S[0] = X_S[1]
            Y_S[0] = Y_S[1]
            X_S[1] = i
            gradient = clau_gradient(X, y_train, theta)
            theta = theta - lr * gradient
            cost_2 = 0.5 * loss(X, y_train, theta).T * loss(X, y_train, theta)
            Y_S[1] = cost_2
            plt.plot(X_S, Y_S)
            plt.pause(0.01)

            i = i + 1

            if abs(cost_1 - cost_2) < 1000:
                break

            cost_1 = cost_2
        plt.ioff()
        plt.show()
        self._theta = theta
        self.intercept = self._theta[0]
        self.coef = self._theta[1:]

    # 方法3：通过随机梯度下降法求系数 一次计算一个样本

    def fit_sgd(self, x_train, y_train, lr=0.01, max_iter=1000):
        def clau_gradient(X, y, theta):
            return X * (X.dot(theta) - y) * 2

        X = np.hstack((np.ones(shape=(len(x_train), 1)), x_train))
        theta = np.zeros(X.shape[1])
        i = 0
        while i < max_iter:
            i = i + 1
            for index in range(len(X)):
                gradient = clau_gradient(X[index], y_train[index], theta)
                theta = theta - lr * gradient
        self._theta = theta
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

    def predeict(self, x_test):
        X = np.hstack((np.ones(shape=(len(x_test), 1)), x_test))
        return np.dot(X, self._theta)

    def score(self, y_pre, y_true):
        u = ((y_true - y_pre) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        R2 = (1 - u / v)
        return R2


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # 数据准备
    X = load_boston()['data']
    y = load_boston()['target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

    # 方法一
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pre = lr.predeict(x_test)
    print(lr.score(y_pre, y_test))
    # 方法二 梯度下降法需要进行标准化
    lr = LinearRegression()
    x_train_standard = StandardScaler().fit_transform(x_train)
    x_test_standard = StandardScaler().fit_transform(x_test)
    lr.fit_gd(x_train_standard, y_train)
    y_pre = lr.predeict(x_test_standard)
    print(lr.score(y_pre, y_test))
    # 方法三 梯度下降法需要进行标准化
    lr = LinearRegression()
    x_train_standard = StandardScaler().fit_transform(x_train)
    x_test_standard = StandardScaler().fit_transform(x_test)
    lr.fit_sgd(x_train_standard, y_train)
    y_pre = lr.predeict(x_test_standard)
    print(lr.score(y_pre, y_test))
    #画loss fuction
    lr.fit_gd_plot(x_train_standard, y_train)

"""
0.7054912046322195
0.6844336936743269
0.6154602658061006
"""
