from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(text1, text2, vocab=[], fitur=False, char = False):
    if fitur==True and char==True:
        # print("TT", end=".")
        if type(vocab)==list:
            vocab = " ".join(vocab)
        vocab = list(set(vocab))
        vocab = " ".join(vocab)
        vocab = list(set(vocab))
        if len(vocab)>= 1:
            vectorizer = TfidfVectorizer(vocabulary=vocab, analyzer='char')
        else:
            vectorizer = TfidfVectorizer()
        return [vectorizer.fit_transform([text1, text2]), vectorizer.fit([text1, text2])]
        
    elif fitur==True and char==False:
        # print("TF", end=".")
        if len(vocab)>= 1:
            vectorizer = TfidfVectorizer(vocabulary=vocab)
        else:
            vectorizer = TfidfVectorizer()
        return [vectorizer.fit_transform([text1, text2]), vectorizer.fit([text1, text2])]
    elif fitur==False and char==True:
        # print("FT", end=".")
        vectorizer = TfidfVectorizer(analyzer='char')
        return [vectorizer.fit_transform([text1, text2]), vectorizer.fit([text1, text2])]
    else:
        # print("FF")
        vectorizer = TfidfVectorizer()
        return [vectorizer.fit_transform([text1, text2]), vectorizer.fit([text1, text2])]

def tf_idf_(list_, char = False):
    # fitur = list(set(fitur))
    if char == True:
        vectorizer = TfidfVectorizer(analyzer='char')
    else:
        vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(list_)