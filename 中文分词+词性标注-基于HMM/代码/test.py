# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 11:03:35 2021

@author: wbb
"""

from models import ShortTokenizer
from models import HmmToken
from models import HmmPosTag

Tokenizer = ShortTokenizer.ShortTokenizer()
    #Tokenizer.train('../data/PeopleDaily_Token.txt')
Tokenizer.train('./data/zuiduanfenci.json', trained=True)
print(Tokenizer.Token('迈向充满希望的新世纪'))
print(Tokenizer.Token('１９９７年，是中国发展历史上非常重要的很不平凡的一年。'))

hmm1=HmmToken.Hmm()
hmm1.load('./data/Hmmfenci.json')
hmm = HmmPosTag.HmmPosTag()
#hmm.train("../data/PeopleDaily_clean.txt",save_model=True)
hmm.load('./data/Hmmbiaozhu.json')
c=input('请输入一句话:\n')
b=hmm1.cut(c)
m=Tokenizer.Token(c)
q1=' '.join(b)
q2=' '.join(m)
result1=hmm.predict(q1)
result=hmm.predict(q2)
print('使用最短路径进行分词然后进行标注:\n',result,'\n使用hmm进行分词然后进行标注:\n',result1)
