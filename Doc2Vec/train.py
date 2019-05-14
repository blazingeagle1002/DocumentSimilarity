from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim import corpora
import gensim
import gensim.downloader as api
from gensim.matutils import softcossim
#from gensim import fasttext_model300
from gensim import *
import fasttext
import gensim.downloader as api
#import csv

'''data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]
        '''

data =[]
with open('out_mod.txt','r') as f:
    docs=f.readlines()
    for l in docs:
        st=l.strip('\n')
        if st!='':
            data.append(st)
#print(data)
        
documents=[]

for d in data:
    documents.append(d.split())

dictionary= corpora.Dictionary(documents)
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm =1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    #print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")



model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
print("Model loaded Input the document")
s= input()
s=s.lower()
test_data = word_tokenize(s)

v1 = model.infer_vector(test_data)
print("V1_infer", v1)

#s = dictionary.doc2bow(s)

max=0
idx=0

for d in data:
    if softcossim(dictionary.doc2bow((s.lower()).split()), dictionary.doc2bow((d.lower()).split()), similarity_matrix) > max:
        max=  softcossim(dictionary.doc2bow((s.lower()).split()), dictionary.doc2bow((d.lower()).split()), similarity_matrix)
        idx= data.index(d)

print(max)
print(data[idx])



# to find most similar doc using tags
#similar_doc = model.docvecs.most_similar('1')

#print("work")
#print(similar_doc)

