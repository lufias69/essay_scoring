import re
import os
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
factoryStop = StopWordRemoverFactory()
stemmer = factory.create_stemmer()
stopword = factoryStop.create_stop_word_remover()

def get_data(name):
    with open(dir_path+"/"+str(name), "r") as filename:
        return json.load(filename)
def save_data(name, data):
    with open(dir_path+"/"+str(name), "w") as filename:
        json.dump(data, filename)

las_use    = get_data("data_stemmer/last_use.json")
fail_stem  = get_data("data_stemmer/fail_stem.json")
kata_dasar = get_data("data_stemmer/kata-dasar.json")

def stemmer_kata(teks):
    # print(len(last_use_k))
    # print(len(last_use_r))
    teks_s = str(teks).split()
    for i, kt in enumerate(teks_s):
        if kt in las_use:
            teks_s[i] = las_use[kt]
            # teks_s[i] = last_use_r[last_use_k.index(kt)]
        elif kt in las_use or kt in fail_stem or kt in kata_dasar:
            continue
        else:
            teks_s[i] = stemmer.stem(kt)
            # print("*", end="")
            if teks_s[i] != kt:
                las_use[kt] = teks_s[i]
                # last_use_k.append(kt)
                # last_use_r.append(teks_s[i])
            else:
                if kt not in fail_stem:
                    # last_use_aneh.append(kt)
                    fail_stem.insert(0,kt)
    try:
        save_data(nama = "data_stemmer/last_use.json",  data=las_use)
        save_data(nama = "data_stemmer/fail_stem.json", data=fail_stem)
    except:
        pass
    return " ".join(teks_s)

def stop_word(kata):
    kata = kata.split()
    n_kata = list()
    for i in kata:
        n_kata.append(stopword.remove(i))
    a = re.sub(' +', ' '," ".join(n_kata))
    a = a.lstrip()
    return a