import os
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import re


dir_path = os.path.dirname(os.path.realpath(__file__))
print("ok")
def stopword():
    with open(dir_path+"/data/id.stopwords.02.01.2016_2.txt") as f:
        data =  f.readlines()
    n_data = list()
    for i in data:
        a = i.replace("\n", "")
        if "-"in a:
            for j in a.split("-"):
                n_data.append(j.replace("\n", ""))
        else:
            n_data.append(i.replace("\n", ""))
    del data
    return list(set(n_data))

def stopwords(kalimat, corpus = stopword()):
    kalimat = tokenizer.tokenize(kalimat)
    for ix, k in enumerate(kalimat):
        if k in corpus:
            kalimat[ix]=""
    kalimat = " ".join(kalimat).lstrip().rstrip()
    return re.sub(' +', ' ',kalimat)

