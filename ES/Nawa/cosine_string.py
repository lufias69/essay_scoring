import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import tf_idf as w
import similarity_ as simi


import math

def Average(lst): 
    return sum(lst) / len(lst) 
def norm (a):
    ls = list()
    for i in a:
        ls.append(i/math.sqrt(sum(a)))
    return ls

def multiplyList(myList) : 
      
    # Multiply elements one by one 
    result = 1
    for x in myList: 
         result = result * x  
    return result  
def idx (i, ax):
    # i = "a"
    # a = "kasura"
    index = list()
    for ix, j in enumerate(ax):
        if j == i:
            index.append(ix+1)
    sum_ = sum(index)
    # avg_ = Average(index)
    # mul = multiplyList(index)
    # return sum_/math.sqrt(math.log(mul,2))
    # return sum_/math.log(mul,2)
    return math.log(sum_)/math.log(math.sqrt(mul),2)
    # return avg_/math.log(mul,2)
    # return sum_


def perpindahan(a,b):
    # a = "kasuraa"
    # b = "rusak"

    ab = list(set(a+b))
    ls_a = list()
    ls_b = list()
    for i in ab:
        try:
            if a.count(i) > 1:
                ls_a.append(idx (i, a))
            else:
                ls_a.append(a.index(i)+1)
        except:
            ls_a.append(0)

        try:
            if b.count(i) > 1:
                ls_b.append(idx (i, b))
            else:
                ls_b.append(b.index(i)+1)
        except:
            ls_b.append(0)
    # ls_a = norm (ls_a)
    # ls_b = norm (ls_b)
    return [ls_a, ls_b]

def cosine_string(a, b, char=True, move=False):
    if move==False:
        t = w.tf_idf(a, b,char = char)
        return round(simi.cosine_similarity(t.A[0], t.A[1]),15)
    else:
        t = w.tf_idf(a, b,char = char)
        cosine = round(simi.cosine_similarity(t.A[0], t.A[1]),15)
        p = perpindahan(a,b)
        pindah = round(simi.cosine_similarity(p[0], p[1]),15)
        return (cosine+pindah)/2
