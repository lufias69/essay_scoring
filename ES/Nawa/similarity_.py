# from math import*
from decimal import Decimal
import numpy as np
from collections import Counter
from math import sqrt

def square_rooted(x):
    return sqrt(sum([a * a for a in x]))
def cosine_similarity(x, y):
    if sum(y) == 0:
        return 0.0
    else:
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = square_rooted(x) * square_rooted(y)
        hasil = numerator / float(denominator)
        return round(hasil,15)

def jaccard(x, y):
    numerator = [a * b for a, b in zip(x, y)]
    x_ = sum(x**2)#x**2
    y_ = sum(y**2)#
    denominator =  (x_ + y_ )- sum(numerator)
    return np.array(sum(numerator)) / np.array(denominator)


def dice_similarity(x, y):
    numerator = [a * b for a, b in zip(x, y)]
    x_ = sum(x**2)
    y_ = sum(y**2)
    denominator =  x_ + y_
    return np.array(2*(sum(numerator))) / np.array(denominator)

def word2vec(word):
    from collections import Counter
    from math import sqrt

    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))
    # return a tuple
    return [cw, sw, lw]

def cosin(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]

def cosine_sim(kj,j):
    vj = word2vec(j)
    vkj = word2vec(kj)
    return cosin(vj,vkj)
