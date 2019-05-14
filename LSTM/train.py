import nltk
from models import InferSent

from random import randint
import numpy as np
import torch

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def max_similar(input,embeddings, tembeddings):
	#max=-1
	li=[]
	for i in range(tembeddings.shape[0]):
		cos=cosine(embeddings[input],tembeddings[i])
		v=(i,cos)
		li.append(v)

	pi= sorted(li, key = lambda x: x[1], reverse= True)[:5]		
	return pi

V = 2
MODEL_PATH = 'encoder/infersent%s.pickle' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = 'dataset/fastText/crawl-300d-2M.vec/crawl-300d-2M-subword.vec'
model.set_w2v_path(W2V_PATH)


use_cuda = False
model = model.cuda() if use_cuda else model

model.build_vocab_k_words(K=500000)


sentences = []
with open('perfect_test.txt') as f:
    for line in f:
        sentences.append(line.strip())

print("The length of test set is")
print()
print(len(sentences))

f.close()


esentences= []

with open('perfect_train.txt') as g:
    for line in g:
        esentences.append(line.strip())

print("The length of embedded set or train set is")
print()
print(len(esentences))

print()

print("embedding starts")

embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)

np.save('embeddings',embeddings)

print("test set embedding done" + '\n')

tembeddings = model.encode(esentences, bsize=128, tokenize=False, verbose=True)

np.save('tembeddings', tembeddings)

print("train set embedding done" + '\n')


#print('nb sentences encoded : {0}'.format(len(embeddings)))

#print(embeddings.shape)
'''input =9;

print("input is " )
print()
print()
print(sentences[input])


print("top is ")'''

for input in range(len(sentences)):

	top = max_similar(input,embeddings, tembeddings)

	#print(top);


	#print("output is")
	#print()
	fo = open("./output/inference_outputs", "a+")

	#print(sentences[top[i][0]])
	fo.write(esentences[top[0][0]]+'\n')
	
	fa=open("./output/output_nos", "a+")
	fa.write("%d" %(top[0][0]+1) + '\n')

	fb=open("./output/sim_scores", "a+")
	fb.write("%f" %(top[0][1]) + '\n')
	
	#print()
fo.close()
fa.close()	
fb.close()

print(sentences[max_similar(input,embeddings)], cosine(embeddings[input],embeddings[max_similar(input,embeddings)]));



for i in range(9):
   _, _ = model.visualize(sentences[i] ,i)

