# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os as os

mpl.rcParams['font.sans-serif'] = ["Droid Sans Fallback"]
mpl.rcParams['axes.unicode_minus'] = False 

file1 = open("shell.txt")
head=[[],[],[],[],[],[],[],[]]
columnss=[]
indexx=[]
context=True
wordnum=0
for line in file1:
    line = line.split()
    if context==True and line[0] != "target:":
        print line[0]
        columnss.append("".join(w for w in line[:]))
    elif line[0]=="target:":
        context=False
        indexx = [w for w in line[1:]]
    else:
        ww = line[0]
        for ii in range(8):
            line=file1.next()
            head[ii].append([float(ff) for ff in line.split()])
        wordnum+=1
        if wordnum==len(indexx):
            context=True
            for h in range(8):
                train_df=head[h]
                df2 = pd.DataFrame(data=train_df, index=indexx, columns=columnss)
                df2.to_csv("testfoo.csv" , encoding = "utf-8")
                df = pd.read_csv("testfoo.csv" , encoding = "utf-8",index_col=0)
                plt.figure(figsize=(25,25))
                title = "</d> ".join(w for w in df.columns)
                plt.title(title, y=1.05, size=15)
                g = sns.heatmap(df)
                plt.xticks(rotation=20) 
                plt.yticks(rotation=360)
                plt.savefig('heatmap'+str(h)+'.png')
            head=[[],[],[],[],[],[],[],[]]
            columnss=[]
            indexx=[]
            wordnum=0
            break
