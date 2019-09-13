import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import cosine_string as cs
from pyjarowinkler import distance
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def ubah_simbol(teks):
    teks = teks.replace(".", " ").replace("}", "").replace("{", "").replace("(", "").replace(")", "").replace("-", "").replace(":", " ").replace(",", " ").replace("!", " ").replace(";", " ").replace("'", " ").replace('"', " ")
    return re.sub(' +', ' ',teks)

def cek_typo(kunci_jawaban, jawaban, toleransi=0.95):
  jawaban =  ubah_simbol(jawaban).lower()
  kunci_jawaban = ubah_simbol(kunci_jawaban).lower()
  kunci_jawaban_split = kunci_jawaban.split()
  jawaban_split = jawaban.split()
  n_jawaban = jawaban_split
  for i in range(len(jawaban_split)):
    w_1 = []
    n_jawaban = jawaban_split
    kunci_jawaban_ = []
    for j in kunci_jawaban_split:
        w_1.append(distance.get_jaro_distance(jawaban_split[i], j, winkler=True, scaling=0.1))
        #         w_1.append(cs.cosine_string(jawaban_split[i], j))
        kunci_jawaban_.append(j)
    if round(max(w_1),4) != 1.0 and max(w_1) > toleransi:
      index = w_1.index(max(w_1))
      n_jawaban[i]= kunci_jawaban_[index]
  return " ".join(n_jawaban)

def cek_negasi(kata_negasi, kata_dicari):
    if type(kata_negasi) != list:
        kata_negasi = [kata_negasi]  
    n_index = []
    for i in kata_negasi:
        index_replace = [(m.end(0)) for m in re.finditer(i,kata_dicari)]
        n_index += index_replace
        #print(n_index)
    for rep in n_index:
        if rep != len(kata_dicari):
            huruf = [x for x in kata_dicari]
            huruf[rep] = "_"
            kata_dicari = "".join(huruf)
    return kata_dicari



# def cek_negasi(kata_negasi, kata_dicari):
#     if type(kata_negasi) != list:
#         kata_negasi = [kata_negasi]  
#     n_index = []
#     for i in kata_negasi:
#         index_replace = [(m.end(0)) for m in re.finditer(i,kata_dicari)]
#         n_index += index_replace
#         #print(n_index)
#     for rep in n_index:
#         if rep != len(kata_dicari) and kata_dicari[rep]==" ":
#             huruf = [x for x in kata_dicari]
#             huruf[rep] = "_"
#             kata_dicari = "".join(huruf)
#     return kata_dicari

def pisahKata(kunci_jawaban, jawaban):
    d_index = []
    b_index = []
    
    for i in kunci_jawaban.split():
        index_replace = [(m.start(0)) for m in re.finditer(i,jawaban)]
        d_index += index_replace
    
        index_replace = [(m.end(0)) for m in re.finditer(i,jawaban)]
        b_index += index_replace
    jawaban_list = [x for x in jawaban]
    for d, b in zip(d_index, b_index):
        #print(d,b)
        jawaban_list[d]= " "+jawaban_list[d]
        jawaban_list[b-1]=jawaban_list[b-1]+" "

    jawaban_list = re.sub(r"\s+", " ","".join(jawaban_list).rstrip().strip().lstrip())
    return jawaban_list
