import gensim
import numpy as np

from time import time
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

start_time = time()
with open("../Data/temp_train", "r") as f:
    train = f.read().split("\n")

with open("../Data/temp_test", "r") as f:
    test = f.read().split("\n")

print("Number of train docs : {}".format(len(train)))
train_docs = [[w.lower() for w in word_tokenize(text)] for text in train]
train_dictionary = gensim.corpora.Dictionary(train_docs)

print("Number of words in train dictionary : {}".format(len(train_dictionary)))

train_corpus = [train_dictionary.doc2bow(gen_doc) for gen_doc in train_docs]

print("Creating TF-IDF model")
tf_idf = gensim.models.TfidfModel(train_corpus)

sims = gensim.similarities.Similarity('../Model/', tf_idf[train_corpus], num_features=len(train_dictionary))
print(sims)

print("Checking for test documents")
print("Number of test docs : {}".format(len(test)))

def check_test(test_text, dictionary, tf_idf, sims):
    query_doc = [w.lower() for w in word_tokenize(test_text)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    result = sims[query_doc_tf_idf]
    return result

op = open("outputs_tfidf.txt","w")
for text in test:
    result = check_test(test_text=text, dictionary=train_dictionary, tf_idf=tf_idf, sims=sims)
    indices = np.asarray(result).argsort()[-5:][::-1]
    print(indices, [result[_] for _ in indices])
    op.write(train[indices[0]] + "\n")

print("Time taken : {}".format(time() - start_time))
