"""
 Env: /anaconda3/python3.7
 Time: 2021/12/27 15:39
 Author: karlieswfit
 File: movie_analysis.py
 Describe: 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud


sns.set()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#导入数据
data=pd.read_csv('./movie.csv',encoding='gb18030')
# data.head()

#通过sort_values函数进行按值（按account票房）排序
top_10_account=data.sort_values(by='account',ascending=False).head(10)
# top_10_account
#对票房前十名的电影可视化

plt.figure(figsize=(12,7))
plt.xticks(rotation=-20)
plt.title("票房前十名的电影")
g=sns.barplot(x="title",y="account",data=top_10_account)
for i in range(len(top_10_account)):
    g.text(i,top_10_account.iloc[i,-4],top_10_account.iloc[i,-4],color="black",ha="center")
plt.savefig('./票房前十名的电影.png')


#评分前十名的电影
#通过sort_values函数进行按值（按score 评分）排序
top_10_score=data.sort_values(by='score',ascending=False).head(10)
top_10_score

plt.figure(figsize=(12,8))
plt.title("评分前十名的电影")
g = sns.barplot(y="title",x="score",data=top_10_score)
for i in range(len(top_10_score)):
    g.text(top_10_score.iloc[i,2],i,top_10_score.iloc[i,2],color="black",ha="center")
plt.savefig('./评分前十名的电影.png')


# 评分前十名的导演
#通过sort_values函数进行按值（按score 评分）排序，然后获取导演的名字
top_10_score=data.sort_values(by='score',ascending=False).head(10)

#评分前十的电影导演
plt.figure(figsize=(12,8))
plt.title("评分前十电影的导演")
g = sns.barplot(y="drector",x="score",data=top_10_score)
for i in range(len(top_10_score)):
    g.text(top_10_score.iloc[i,2],i,top_10_score.iloc[i,2],color="black",ha="center")
plt.savefig('./评分前十电影的导演.png')

#词图
w = wordcloud.WordCloud(width=700, height=400,
               background_color='white',
               max_words=500,
               font_path='./FZPTYJW.TTF',
               max_font_size=66,
               relative_scaling=0.6,  # 设置字体大小与词频的关联程度为0.4
               random_state=50,
               scale=2
               )
data1=data.head(2000)


title=data1["title"]
score=data1["score"]
dic = dict(zip(title, score))

w.generate_from_frequencies(frequencies=dic)
w.to_file('./云图.png')
plt.imshow(w)
plt.show()




