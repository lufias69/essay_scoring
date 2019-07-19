from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(text1, text2, fitur):
    fitur = list(set(fitur))
    vectorizer = TfidfVectorizer(vocabulary=fitur)
    tfidf = vectorizer.fit_transform([text1, text2])
    return tfidf

def tf_idf_(list_):
    if type(list_)!=list:
        list_ = [list_]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(list_)
    return tfidf.A