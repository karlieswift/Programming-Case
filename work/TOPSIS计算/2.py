import pandas as pd
import numpy as np
data = pd.read_excel("work.xlsx")

data=data.T
data=data.reset_index()
data=pd.DataFrame(data=data.values[1:],columns=data.values[0])
data.set_index(['属性(指标)'], inplace=True)

m, n = data.shape  # 获取行数m和列数n
# 熵权法计算
def My_Stander(data1):  # 矩阵标准化(min-max标准化)
    for i in data1.columns:
        for j in range(n+1):
            if i == str(f'X{j}负'):  # 负向指标
                data1[i] = (np.max(data1[i])-data1[i]) / \
                    (np.max(data1[i])-np.min(data1[i]))
            else:  # 正向指标
                data1[i] = (data1[i]-np.min(data1[i])) / \
                    (np.max(data1[i])-np.min(data1[i]))
    return data1
stander_data= My_Stander(data)  # 标准化矩阵

formater="{0:.04f}".format
stander_data=stander_data.applymap(formater)



None_ij = [[None] * n for i in range(m)]  # 新建空矩阵
def entropy(data):  # 计算熵值
    data = np.array(data,dtype='float')
    E = np.array(None_ij)
    for i in range(m):
        for j in range(n):
            if data[i][j] == 0:
                e_ij = 0.0
            else:
                P_ij = data[i][j] / data.sum(axis=0)[j]  # 计算比重
                e_ij = (-1 / np.log(m)) * P_ij * np.log(P_ij)
            E[i][j] = e_ij
    E_j = E.sum(axis=0)
    G_j = 1 - E_j
    W_j = G_j / sum(G_j)
    return np.around(np.array(W_j,dtype="float32"),1)
W = entropy(stander_data)

stander_data=stander_data.T
save_stander_data=stander_data.copy()
save_stander_data['权重']=W
save_stander_data.to_excel('./标准化后的数据.xlsx')


W_j = W
# TOPSIS计算
stan_data = np.array(stander_data.T,dtype='float')  # stan_data为标准化矩阵
Z_ij = np.array(None_ij)  # 空矩阵
for i in range(m):
    for j in range(n):
        Z_ij[i][j] = stan_data[i][j]*W_j[j]  # 计算加权标准化矩阵Z_ij
Imax_j = Z_ij.max(axis=0)  # 最优解
Imin_j = Z_ij.min(axis=0)  # 最劣解
Dmax_ij = np.array(None_ij)  # 空矩阵
Dmin_ij = np.array(None_ij)  # 空矩阵
for i in range(m):
    for j in range(n):
        Dmax_ij[i][j] = (Imax_j[j] - Z_ij[i][j]) ** 2
        Dmin_ij[i][j] = (Imin_j[j] - Z_ij[i][j]) ** 2
Dmax_i = Dmax_ij.sum(axis=1)**0.5  # 最优解欧氏距离
Dmin_i = Dmin_ij.sum(axis=1)**0.5  # 最劣解欧氏距离
score = Dmin_i/(Dmax_i+Dmin_i)  # 综合评价值
result=pd.DataFrame(stander_data.T.index)
result['TOPSIS法得分']=score

result=result.sort_values("TOPSIS法得分",ascending=0)

result['TOPSIS法排名']=[i+1 for i in range(len(result))]


result.to_excel('./TOPSIS法排名.xlsx')

print(result)
