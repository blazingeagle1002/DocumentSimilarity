import gensim
import numpy as np

from time import time
from nltk.tokenize import word_tokenize

start_time = time()

with open("../Data/perfect_train.txt", "r") as f:
    train = f.read().split("\n")

with open("../Data/perfect_test.txt", "r") as f:
    test = f.read().split("\n")

print("Number of train docs : {}".format(len(train)))
train_docs = [[w.lower() for w in word_tokenize(text)] for text in train]
train_dictionary = gensim.corpora.Dictionary(train_docs)

print("Number of words in train dictionary : {}".format(len(train_dictionary)))

train_corpus = [train_dictionary.doc2bow(gen_doc) for gen_doc in train_docs]

print("Creating LDA model")
lda_model = gensim.models.LdaModel(train_corpus, num_topics = 500)

sims = gensim.similarities.Similarity('../Model/', lda_model[train_corpus], num_features=len(train_dictionary))
print(sims)

print("Checking for test documents")
print("Number of test docs : {}".format(len(test)))
op = open("outs_lda.txt", "w")
op2 = open("test_file_lda.txt","w")

def check_test(test_text, dictionary, lda_model, sims):
    query_doc = [w.lower() for w in word_tokenize(test_text)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = lda_model[query_doc_bow]
    result = sims[query_doc_tf_idf]
    return result

for text in test:
    result = check_test(test_text=text, dictionary=train_dictionary, lda_model=lda_model, sims=sims)
    indices = np.asarray(result).argsort()[-5:][::-1]
    print(indices, [result[_] for _ in indices])
    op.write(str(indices[0]) + "\t" + str(result[indices[0]]) + "\t" + train[indices[0]] + "\n")
    op2.write(text + "\n")

print("Time taken : {}".format(time() - start_time))
