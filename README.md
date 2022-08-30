# DEPA_text_analyse
# 案例：量化DEPA规则，提取词频
## 1.导入包
import numpy as np  
import pandas as pd  
import jieba  
import jieba.analyse  
import matplotlib.pyplot as plt  
import wordcloud  
## 2.自定义词典 & 使用withopen导入文本
jieba.load_userdict('data/custom.txt')
with open('data/DEPA.txt','r',encoding='utf-8') as f:
    a = f.read()
    
## 3.分词
cut = jieba.lcut(a)
## 4.去停用词
stopword=[]  
with open('data/stopword.txt','r',encoding='utf-8') as f :  
    for line in f.readlines():  
        l = line.strip()  
        if l == '\\n':  #换行符  
            l = '\n'  
        if l == '\\u3000' : #制表符  
            l = '\u3000'  
        stopword.append(l)  
#去停用词 第一步  
x = np.array(cut)    #将分好的此列表转为数组  
y = np.array(stopword)   #将停用词转为数组  
z = x[~np.in1d(x,y)]  
#去停用词 第二部  
k = [i for i in z if len(i)>1 ]  

## 5.计算词频，排序
result = pd.DataFrame(k).groupby(0).size().sort_values(ascending=False) [:20]
#result

## 6.输出关键词
result.to_csv('tmp/DEPA_keyword_fig.csv',header=False,encoding='GBK')

## 7.制作词云图
import matplotlib.pyplot as plt  
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator  
  
#backgroup_Image = plt.imread('F:/man.jpg') #笼罩图  
  
f = open('tmp/DEPA_keyword_fig.csv','r').read()  #生成词云的文档  
wordcloud = WordCloud(  
        background_color = 'white', #背景颜色，根据图片背景设置，默认为黑色  
        #mask = backgroup_Image, #笼罩图  
        font_path = 'C:\Windows\Fonts\STZHONGS.TTF',#若有中文需要设置才会显示中文  
        width = 1000,  
        height = 860,  
        margin = 2).generate(f) # generate 可以对全部文本进行自动分词  
#参数 width，height，margin分别对应宽度像素，长度像素，边缘空白处  
  
plt.imshow(wordcloud)  
plt.axis('off')  
plt.show()  
  
#保存图片：默认为此代码保存的路径  
wordcloud.to_file('DEPA.jpg') 

![image](https://github.com/mengke25/DEPA_text_analyse/blob/main/DEPA.jpg)
