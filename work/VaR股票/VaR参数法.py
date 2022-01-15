"""
 Env: /anaconda3/python3.7
 Time: 2021/12/28 12:34
 Author: karlieswfit
 File: VaR参数法.py
 Describe: 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


data=pd.read_csv('./data.csv')

# 股票的开盘收盘价走势：
plt.figure(figsize=(12,6))
plt.plot(data['开盘'],label="open")
plt.plot(data['收盘'],label="close")
plt.xlabel("February 2019 to March 2021")
plt.ylabel("open/close")
plt.title("Opening and closing price trend of stocks from February 2019 to March 2021")
plt.legend()
plt.savefig('./2019年2月到2021年3月股票的开盘收盘价走势.png')
plt.show()


# #计算每天的收益率
data['收益率'] = data['收盘'].pct_change()


mean_return = data['收益率'].mean()
std = data['收益率'].std()
VaR_1 = mean_return-2.33*std
VaR_5 = mean_return-1.645*std
print("VaR_1:",VaR_1)
print("VaR_5:",VaR_5)

file = open(r'./VaR.txt', mode='w')
file.write("VaR_1:{0}".format(VaR_1))  #将abc写入 txt文件中
file.write('\n')      #将回车写入txt文件中
file.write("VaR_5:{0}".format(VaR_5))  #将abc写入 txt文件中
file.close()








plt.figure(figsize= (10,6))
plt.title("Returns and normal distribution")
sns.distplot(data["收益率"].dropna(),bins=50)
plt.xlabel('returns')
plt.ylabel('count')
plt.savefig('./收益和正态分布1.png')
plt.show()


plt.figure(figsize= (10,6))
plt.title("Returns and normal distribution")
plt.axvline(VaR_5, color='r', linewidth=1,label = 'VaR_5')
plt.axvline(VaR_1, color='y', linewidth=1,label = 'VaR_1')
plt.legend()
sns.distplot(data["收益率"].dropna(),bins=50)
plt.xlabel('returns')
plt.ylabel('count')
plt.savefig('./收益和正态分布2.png')
plt.show()


