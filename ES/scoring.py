
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import pandas as pd
from Nawa import similarity_ as simi
from Nawa import tf_idf as w
from Nawa import ngram
from Nawa import normalisasi as nrm

# fitur =list("abcdefghijklmnopqrstufwxyz 1234567890")

def t2c(text):
    lst = list(text)
    return " ".join(lst)

def transform(simmm):
    if simmm <= 0.50:
        return "sangat tidak mirip"
    elif simmm <= 0.75:
        return "tidak mirip"
    elif simmm <= 0.80:
        return "mirip"
    elif simmm <= 1.0002:
        return "sangat mirip"
    else:
        return "error"

def cek_mised(data):
    # print(data)
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
        # hasil = nrm.cek_negasi(kata_negasi,i)
        new_kj.append(nrm.cek_negasi(kata_negasi,i))
    return new_kj

def get_unik(kunci_jawaban):
    if type(kunci_jawaban) is not list:
        kunci_jawaban = [kunci_jawaban]
    list_kata_kj = list()
    for i in kunci_jawaban:
        i = nrm.ubah_simbol(i)
        for j in i.split():
            list_kata_kj.append(j)
    # list_kata_kj = list(set(list_kata_kj))
    return list(set(list_kata_kj))
    
def praproses(jawaban, kunci_jawaban_unik):
    jawaban = nrm.ubah_simbol(jawaban)
    # jawaban = nrm.pisahKata(kunci_jawaban_unik, jawaban)
    jawaban = nrm.cek_typo(kunci_jawaban_unik, jawaban, 0.96)
    jawaban_ngram = ngram.en_geram(kunci_jawaban_unik, jawaban).split()
    # print("n-gram",jawaban_ngram)
    # jawaban = jawaban +" "+ " ".join(list(set(jawaban_ngram.split())))
    jawaban = jawaban.split()
    for i in jawaban_ngram:
        if i not in jawaban:
            jawaban.append(i)
    jawaban = " ".join(jawaban)
    # print(jawaban)
    return nrm.cek_negasi(kata_negasi, jawaban)

def essay_jaccard_similarity(kunci_jawaban, jawaban, char=False, fitur=False):
    if type(kunci_jawaban) != list:
        kunci_jawaban = [kunci_jawaban]
    kunci_jawaban = cek_negasi_list(kunci_jawaban)
    kunci_jawaban_unik = " ".join(get_unik(kunci_jawaban))
    # print(kunci_jawaban_unik)
    jawaban = praproses(jawaban, kunci_jawaban_unik)
    # print(jawaban)
    simm = list()   
    for kj in kunci_jawaban:
        fitur_ = list(set(kj.split()))
        t = w.tf_idf(kj, jawaban, vocab=fitur_, fitur=fitur, char = char)
        # print(t.A)
        jaccard = simi.jaccard(t.A[0], t.A[1])
        simm.append(jaccard)
    return [round(max(simm),10), transform(max(simm))]

def essay_cosine_similarity(kunci_jawaban, jawaban, char=False, fitur=False):
    if type(kunci_jawaban) != list:
        kunci_jawaban = [kunci_jawaban]
    kunci_jawaban = cek_negasi_list(kunci_jawaban)
    kunci_jawaban_unik = " ".join(get_unik(kunci_jawaban))
    # print(kunci_jawaban_unik)
    jawaban = praproses(jawaban, kunci_jawaban_unik)
    # print(jawaban)
    simm = list()   
    for kj in kunci_jawaban:
        fitur_ = list(set(kj.split()))
        t = w.tf_idf(kj, jawaban, vocab=fitur_, fitur=fitur, char = char)
        # print(t.A)
        cosine = simi.cosine_similarity(t.A[0], t.A[1])
        simm.append(cosine)
    return [round(max(simm),10), transform(max(simm))]


def essay_dice_similarity(kunci_jawaban, jawaban, char=False, fitur=False):
    if type(kunci_jawaban) != list:
        kunci_jawaban = [kunci_jawaban]
    kunci_jawaban = cek_negasi_list(kunci_jawaban)
    kunci_jawaban_unik = " ".join(get_unik(kunci_jawaban))
    # print(kunci_jawaban_unik)
    jawaban = praproses(jawaban, kunci_jawaban_unik)
    # print(jawaban)
    simm = list()   
    for kj in kunci_jawaban:
        fitur_ = list(set(kj.split()))
        t = w.tf_idf(kj, jawaban, vocab=fitur_, fitur=fitur, char = char)
        # print(t.A)
        dice = simi.dice_similarity(t.A[0], t.A[1])
        simm.append(dice)
    return [round(max(simm),10), transform(max(simm))]

def scoring_cosine(jawaban, kuncijawaban_list, skor_list, char=False, fitur=False):
    ratio=list()
    tranform = list()
    for i in kuncijawaban_list:
        # rat = fuzz.token_set_ratio(i, gejala)
        # rat = essay_cosine_similarity(i, jawaban, bychar, fixed)
        rat =  essay_cosine_similarity(i, jawaban, fitur=fitur, char = char)
        ratio.append(rat[0])
        tranform.append(rat[1])

    dict_ = {
        "skor":skor_list,
        "similarity":ratio,
        "transform":tranform,
        # "keterangan":keterangan
    }
    dataf = pd.DataFrame.from_dict(dict_)
    sorted_ =  dataf.sort_values(by=['similarity'], ascending=False)
    # print(sorted_)
    skor = sorted_['skor'].tolist()[0]
    similarity = round(sorted_['similarity'].tolist()[0],3)
    # ket = sorted_['keterangan'].tolist()[0]
    tr = sorted_['transform'].tolist()[0]

    return [skor, similarity, tr, sorted_]
    # return {"skor":skor, "similarity":similarity, "transform":tr, "sorted":sorted_}

def scoring_jaccard(jawaban, kuncijawaban_list, skor_list, char=False, fitur=False):
    ratio=list()
    tranform = list()
    for i in kuncijawaban_list:
        # rat = fuzz.token_set_ratio(i, gejala)
        rat = essay_jaccard_similarity(i, jawaban, fitur=fitur, char = char)
        # rat = essay_jaccard_similarity(i, jawaban)
        ratio.append(rat[0])
        tranform.append(rat[1])

    dict_ = {
        "skor":skor_list,
        "similarity":ratio,
        "transform":tranform,
        # "keterangan":keterangan
    }
    dataf = pd.DataFrame.from_dict(dict_)
    sorted_ =  dataf.sort_values(by=['similarity'], ascending=False)
    # print(sorted_)
    skor = sorted_['skor'].tolist()[0]
    similarity = round(sorted_['similarity'].tolist()[0],3)
    # ket = sorted_['keterangan'].tolist()[0]
    tr = sorted_['transform'].tolist()[0]
    return [skor, similarity, tr, sorted_]
    # return {"skor":skor, "similarity":similarity, "transform":tr, "sorted":sorted_}

def scoring_dice(jawaban, kuncijawaban_list, skor_list,  char=False, fitur=False):
    ratio=list()
    tranform = list()
    for i in kuncijawaban_list:
        # rat = fuzz.token_set_ratio(i, gejala)
        # rat = essay_dice_similarity(i, jawaban)
        rat = essay_dice_similarity(i, jawaban, fitur=fitur, char = char)
        ratio.append(rat[0])
        tranform.append(rat[1])

    dict_ = {
        "skor":skor_list,
        "similarity":ratio,
        "transform":tranform,
        # "keterangan":keterangan
    }
    dataf = pd.DataFrame.from_dict(dict_)
    sorted_ =  dataf.sort_values(by=['similarity'], ascending=False)
    # print(sorted_)
    skor = sorted_['skor'].tolist()[0]
    similarity = round(sorted_['similarity'].tolist()[0],3)
    # ket = sorted_['keterangan'].tolist()[0]
    tr = sorted_['transform'].tolist()[0]
  
    return [skor, similarity, tr, sorted_]
    # return {"skor":skor, "similarity":similarity, "transform":tr, "sorted":sorted_}

def seleksi_kunci_jawaban(df_kj):
    kuncijawaban_list = df_kj["kunci jawaban"].tolist()
    skor_list = df_kj["skor"].tolist()
    # keterangan = df_kj["keterangan"].tolist()

    kuncijawaban = w.tf_idf_(kuncijawaban_list).A

    kuncijawaban_n = list()
    kuncijawaban_n2 = list()
    skor_n = list()
    # keterangan_n = list()
    i = 0
    for kj, skr, old in zip(kuncijawaban, skor_list, kuncijawaban_list):
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
            # keterangan_n.append(ket)
            
        if flag == True:
            kuncijawaban_n.append(kj)
            kuncijawaban_n2.append(old)
            skor_n.append(skr)
            # keterangan_n.append(ket)
        i+=1
    return {"kunci_jawaban":kuncijawaban_n2, "skor":skor_n}


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

def seleksi_data(respon, label, char = False, batas = .96):
    kuncijawaban_list = respon
    skor_list = label
    # keterangan = df_kj["keterangan"].tolist() char = False

    kuncijawaban = w.tf_idf_(kuncijawaban_list, char = char).A

    kuncijawaban_n = list()
    kuncijawaban_n2 = list()
    skor_n = list()
    # keterangan_n = list()
    i = 0
    for kj, skr, old in zip(kuncijawaban, skor_list, kuncijawaban_list):
        flag = True
        if i!=0:
            for j in kuncijawaban_n:
                cosine = simi.cosine_similarity(j, kj)
                # print(cosine)
                if cosine > batas:
                    # print("lebih")
                    flag = False
                    break
                else:
                    continue
        else:
            kuncijawaban_n.append(kj)
            kuncijawaban_n2.append(old)
            skor_n.append(skr)
            # keterangan_n.append(ket)
            
        if flag == True:
            kuncijawaban_n.append(kj)
            kuncijawaban_n2.append(old)
            skor_n.append(skr)
            # keterangan_n.append(ket)
        i+=1
    return {"respon":kuncijawaban_n2, "label":skor_n}