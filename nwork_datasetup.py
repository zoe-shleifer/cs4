def get_labeled_words(examples=1000):
    '''get_labeled_words(examples=1000) returns X,y of word vecs and labels'''
    import nltk
    import csv
    from keras.preprocessing.text import Tokenizer
    from nltk.corpus import stopwords  
    from nltk.tokenize import word_tokenize
    import pandas as pd
    import numpy as np
    tsv_file = open('/Users/zoeshleifer/Downloads/labeledTrainData.tsv')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    docs = [row[-1] for row in read_tsv][1:examples]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs)
    vocab = list(tokenizer.word_index.keys())
    stop_words = set(stopwords.words('english'))
    bad_words = list(set(vocab).intersection(stop_words))
    display = tokenizer.texts_to_matrix(docs, mode='count')
    df = pd.DataFrame(display).drop([0],axis=1)
    df = df.set_axis(vocab,axis=1)
    df = df.drop(bad_words,axis=1)
    labels = [row[1] for row in read_tsv][1:examples]
    X = np.array(df.astype(float))    
    y = np.array(pd.Series(labels).astype(float))
    return X,y