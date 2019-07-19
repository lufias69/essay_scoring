from math import*
from decimal import Decimal
import numpy as np

def square_rooted(x):
    return sqrt(sum([a * a for a in x]))
def cosine_similarity(x, y):
    if sum(y) == 0:
        return 0.0
    else:
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = square_rooted(x) * square_rooted(y)
        hasil = numerator / float(denominator)
        return hasil

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