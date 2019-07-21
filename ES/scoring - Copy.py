import os
import sys
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from Nawa import similarity as simi
from Nawa import tf_idf as w
from Nawa import ngram
from Nawa import normalisasi as nrm

fitur =list("abcdefghijklmnopqrstufwxyz 1234567890")

def t2c(text):
    lst = list(text)
    return " ".join(lst)

def transform(simmm):
    if simmm <= 0.25:
        return "sangat tidak mirip"
    elif simmm <= 0.5:
        return "tidak mirip"
    elif simmm <= 0.80:
        return "mirip"
    elif simmm <= 1.2:
        return "sangat mirip"
    else:
        return "error"

def cek_mised(data):
    if data == "tidak mirip":
        return "missed"
    elif data == "sangat tidak mirip":
        return "wrong"
    elif data == "mirip" or data == "sangat mirip":
        return "correct"
    else:
        return "error"

kata_negasi = ["tidak", "bukan","tiada","tak", "jangan"]

def cek_negasi_list(kunci_jawaban):
    if type(kunci_jawaban) is not list:
        kunci_jawaban = [kunci_jawaban]
    new_kj = list()
    for i in kunci_jawaban:
        i = nrm.ubah_simbol(i)
        hasil = nrm.cek_negasi(kata_negasi,i)
        new_kj.append(hasil)
    return new_kj
#cek_negasi_list("tidak telepon pada 02173811111 website buanawisata.com")

def get_unik(kunci_jawaban):
    if type(kunci_jawaban) is not list:
        kunci_jawaban = [kunci_jawaban]
    list_kata_kj = list()
    for i in kunci_jawaban:
        i = nrm.ubah_simbol(i)
        for j in i.split():
            list_kata_kj.append(j)
    list_kata_kj = list(set(list_kata_kj))
    return list_kata_kj


def praproses(jawaban, kunci_jawaban_unik):
    jawaban = nrm.ubah_simbol(jawaban)
    jawaban = nrm.pisahKata(kunci_jawaban_unik, jawaban)
    jawaban = nrm.cek_typo(kunci_jawaban_unik, jawaban, 0.95)
    jawaban_ngram = ngram.en_geram(kunci_jawaban_unik, jawaban).split()
    #jawaban = jawaban +" "+ " ".join(list(set(jawaban_ngram.split())))
    jawaban = jawaban.split()
    for i in jawaban_ngram:
        if i not in jawaban:
            jawaban.append(i)
    jawaban = " ".join(jawaban)
    jawaban = nrm.cek_negasi(kata_negasi, jawaban)
    return jawaban



def essay_jaccard_similarity(kunci_jawaban, jawaban, bychar=False, fixed=False):
    if type(kunci_jawaban) != list:
        kunci_jawaban = [kunci_jawaban]
    kunci_jawaban = cek_negasi_list(kunci_jawaban)
    kunci_jawaban_unik = " ".join(get_unik(kunci_jawaban))
    # print(kunci_jawaban_unik)
    jawaban = praproses(jawaban, kunci_jawaban_unik)
    # print(jawaban)
    simm = list()   
    for kj in kunci_jawaban:
        fitur = list(set(kj.split()))
        #print(fitur)
        t = w.tf_idf(kj, jawaban, fitur)
        jaccard = simi.jaccard(t.A[0], t.A[1])
        simm.append(jaccard)
    return [max(simm), transform(max(simm))]

def essay_cosine_similarity(kunci_jawaban, jawaban, char=False, fitur=True):
    if type(kunci_jawaban) != list:
        kunci_jawaban = [kunci_jawaban]
    kunci_jawaban = cek_negasi_list(kunci_jawaban)
    kunci_jawaban_unik = " ".join(get_unik(kunci_jawaban))
    # print(kunci_jawaban_unik)
    jawaban = praproses(jawaban, kunci_jawaban_unik)
    # print(jawaban)
    simm = list()   
    for kj in kunci_jawaban:
        if char == True and fitur==True:
            print("tt",end="")
        elif char == True and fitur==False:
            print("tf",end="")
        elif char == False and fitur==True:
            print("ft",end="")
        elif char == False and fitur==False:
            print("ff",end="")
        fitur_ = list(set(kj.split()))
        t = w.tf_idf(kj, jawaban, vocab=fitur_, fitur=fitur, char = char)
        print(t.A)
        cosine = simi.cosine_similarity(t.A[0], t.A[1])
        simm.append(cosine)
    return [round(max(simm),10), transform(max(simm))]


def essay_dice_similarity(kunci_jawaban, jawaban, bychar=False, fixed=False):
    if type(kunci_jawaban) != list:
        kunci_jawaban = [kunci_jawaban]
    kunci_jawaban = cek_negasi_list(kunci_jawaban)
    kunci_jawaban_unik = " ".join(get_unik(kunci_jawaban))
    # print(kunci_jawaban_unik)
    jawaban = praproses(jawaban, kunci_jawaban_unik)
    # print(jawaban)
    simm = list()   
    for kj in kunci_jawaban:
        # print(t2c(kj))
        t = w.tf_idf(kj, jawaban, fitur)
        # print(t.A)
        dice = simi.dice_similarity(t.A[0], t.A[1])
        print(dice)
        # simm.append(dice)
    # return [max(simm), transform(max(simm))]


def scoring_cosine(jawaban, kuncijawaban_list, skor_list, keterangan, bychar=False, fixed=False):
    ratio=list()
    tranform = list()
    for i in kuncijawaban_list:
        # rat = fuzz.token_set_ratio(i, gejala)
        rat = essay_cosine_similarity(i, jawaban, bychar, fixed)
        ratio.append(rat[0])
        tranform.append(rat[1])

    dict_ = {
        "skor":skor_list,
        "similarity":ratio,
        "transform":tranform,
        "keterangan":keterangan
    }
    dataf = pd.DataFrame.from_dict(dict_)
    sorted_ =  dataf.sort_values(by=['similarity'], ascending=False)
    # print(sorted_)
    skor = sorted_['skor'].tolist()[0]
    similarity = round(sorted_['similarity'].tolist()[0],3)
    ket = sorted_['keterangan'].tolist()[0]
    tr = sorted_['transform'].tolist()[0]
    # print("skor       ",skor)
    # print("similarity ",similarity)
    # print("transform  ",tr)
    return [skor, ket, similarity, tr, sorted_]

def scoring_jaccard(jawaban, kuncijawaban_list, skor_list, keterangan):
    ratio=list()
    tranform = list()
    for i in kuncijawaban_list:
        # rat = fuzz.token_set_ratio(i, gejala)
        rat = essay_jaccard_similarity(i, jawaban)
        ratio.append(rat[0])
        tranform.append(rat[1])

    dict_ = {
        "skor":skor_list,
        "similarity":ratio,
        "transform":tranform,
        "keterangan":keterangan
    }
    dataf = pd.DataFrame.from_dict(dict_)
    sorted_ =  dataf.sort_values(by=['similarity'], ascending=False)
    # print(sorted_)
    skor = sorted_['skor'].tolist()[0]
    similarity = round(sorted_['similarity'].tolist()[0],3)
    ket = sorted_['keterangan'].tolist()[0]
    tr = sorted_['transform'].tolist()[0]
    # print("skor       ",skor)
    # print("similarity ",similarity)
    # print("transform  ",tr)
    return [skor, ket, similarity, tr, sorted_]

def scoring_dice(jawaban, kuncijawaban_list, skor_list, keterangan):
    ratio=list()
    tranform = list()
    for i in kuncijawaban_list:
        # rat = fuzz.token_set_ratio(i, gejala)
        rat = essay_dice_similarity(i, jawaban)
        ratio.append(rat[0])
        tranform.append(rat[1])

    dict_ = {
        "skor":skor_list,
        "similarity":ratio,
        "transform":tranform,
        "keterangan":keterangan
    }
    dataf = pd.DataFrame.from_dict(dict_)
    sorted_ =  dataf.sort_values(by=['similarity'], ascending=False)
    # print(sorted_)
    skor = sorted_['skor'].tolist()[0]
    similarity = round(sorted_['similarity'].tolist()[0],3)
    ket = sorted_['keterangan'].tolist()[0]
    tr = sorted_['transform'].tolist()[0]
    # print("skor       ",skor)
    # print("similarity ",similarity)
    # print("transform  ",tr)
    return [skor, ket, similarity, tr, sorted_]

def seleksi_kunci_jawaban(df_kj):
    kuncijawaban_list = df_kj["kunci jawaban"].tolist()
    skor_list = df_kj["skor"].tolist()
    keterangan = df_kj["keterangan"].tolist()

    kuncijawaban = w.tf_idf_(kuncijawaban_list)

    kuncijawaban_n = list()
    kuncijawaban_n2 = list()
    skor_n = list()
    keterangan_n = list()
    i = 0
    for kj, skr, ket, old in zip(kuncijawaban, skor_list, keterangan, kuncijawaban_list):
        flag = True
        if i!=0:
            for j in kuncijawaban_n:
                cosine = simi.cosine_similarity(j, kj)
                # print(cosine)
                if cosine > .96:
                    # print("lebih")
                    flag = False
                    break
                else:
                    continue
        else:
            kuncijawaban_n.append(kj)
            kuncijawaban_n2.append(old)
            skor_n.append(skr)
            keterangan_n.append(ket)
            
        if flag == True:
            kuncijawaban_n.append(kj)
            kuncijawaban_n2.append(old)
            skor_n.append(skr)
            keterangan_n.append(ket)
        i+=1
    return {"kunci_jawaban":kuncijawaban_n2, "skor":skor_n, "keterangan":keterangan_n}


def cosine_huruf(kunci_jawaban, jawaban):
    if type(kunci_jawaban) != list:
        kunci_jawaban = [kunci_jawaban]
    kunci_jawaban = cek_negasi_list(kunci_jawaban)
    kunci_jawaban_unik = " ".join(get_unik(kunci_jawaban))
    # print(kunci_jawaban_unik)
    jawaban = praproses(jawaban, kunci_jawaban_unik)
    # print(jawaban)
    simm = list()   
    for kj in kunci_jawaban:
        simm.append(simi.cosine_sim(jawaban,kj))
    return [max(simm), transform(max(simm))]

        
        




