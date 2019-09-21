from sklearn.feature_extraction.text import CountVectorizer
import math
# import numpy as np
# corpus = [
#     'makan nasi ayam',
#     'nasi',
# ]
# fitur = ['makan', 'nasi', 'ayam']

def tf_idf (corpus, vocab):
    vectorizer = CountVectorizer(vocabulary = vocab)
    X = vectorizer.fit_transform(corpus)
#     print(vectorizer.get_feature_names())

#     print(X.toarray())
    # X = X.transpose().toarray()
    X = X.toarray()
    # X
    import numpy as np
    import math
    df = list()
#     print(X)
    for i in X.transpose():
    #     print(i)
        count = 0
        for j in i:
            if j>0:
                count+=1
        df.append(count)
#     print("df ",df)
    n_per_df = list()
    for i in df:
        n_per_df.append(len(X)/i) 
#     print("n/df",n_per_df)
    idf_ = list()
    for i in n_per_df:
        idf_.append(math.log(i)+1)
#     print('idf',idf_)
#     print(len(idf_))
    return X*idf_