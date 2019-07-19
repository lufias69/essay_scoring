# from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from pyjarowinkler import distance
import re

def get_ngrams(text, n ):
    n_grams = ngrams(text.split(), n)
    return [ ' '.join(grams) for grams in n_grams]

def toTeks(ng):
    teks_ = []
    for i in ng:
        a = i.split()
        a = "".join(a)
        teks_.append(a)
    return " ".join(teks_)

def hapusKosong(ass):
    apa = []
    for i in ass:
        if i != '':
            apa.append(i)
    return apa
def cek_tipo(kunci_jawaban, jawaban, toleransi=0.97):
    kunci_jawaban_split = kunci_jawaban.split()
    jawaban_split = jawaban.split()
    #n_jawaban = jawaban_split
    for i in range(len(jawaban_split)):
        w_1 = []
        n_jawaban = jawaban_split
        kunci_jawaban_ = []
        for j in kunci_jawaban_split:
            w_1.append(distance.get_jaro_distance(jawaban_split[i], j, winkler=True, scaling=0.1))
            kunci_jawaban_.append(j)
        if max(w_1) >= toleransi:
            index = w_1.index(max(w_1))
            n_jawaban[i]= kunci_jawaban_[index]
        else:
            n_jawaban[i]=""
    x= hapusKosong(n_jawaban)
      
    return " ".join(x)

def pisahKata(dicari, jawaban):
    dicari_split = dicari.split()
    jawaban_split = jawaban.split()
    for h in dicari_split:
        for ct, i in enumerate(jawaban_split) :
            p_dicari = len(h)
            index_depan = 0
            if re.search(h, i):
                index_depan = re.search(h, i).start()
                index_akhir = re.search(h, i).end()
            if len(i) >= p_dicari:
                if re.search(h, i):
                    i = [x for x in i]
                    i.insert(index_akhir, " ")
                    if index_depan != 0:
                        i.insert(index_depan, " ")
                    i = "".join(i)
                    jawaban_split[ct] = i
    return " ".join(jawaban_split)

def en_geram(kunci,teks):
    
    panjangTkes = teks.split()
    #print(len(panjangTkes))
    if len(panjangTkes)<2:
        return teks
    elif len(panjangTkes)<3:
        ng2 = get_ngrams(teks,2)
        tks = ng2
        jawaban = toTeks(tks)
        # return jawaban
        return cek_tipo(kunci, jawaban)
    elif len(panjangTkes)<4:
        ng3 = get_ngrams(teks,3)
        ng2 = get_ngrams(teks,2)
        tks = ng2+ng3
        jawaban = toTeks(tks)
        # return jawaban
        return cek_tipo(kunci, jawaban)
    elif len(panjangTkes) >= 4:
        ng3 = get_ngrams(teks,3)
        ng2 = get_ngrams(teks,2)
        ng4 = get_ngrams(teks,4)
        tks = ng2+ng3+ng4
        jawaban = toTeks(tks)
        # return jawaban
        return cek_tipo(kunci, jawaban)