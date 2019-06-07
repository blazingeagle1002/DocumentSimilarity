import nltk
from nltk.tokenize import word_tokenize
import re

def clean(fp):
    flist = list(filter(None, fp.split("\n")))
    for ind, line in enumerate(flist):
        flist[ind] = line.strip()
    return flist

def create_set(item):
    item = word_tokenize(item)
    for ind, i in enumerate(item):
        item[ind] = i.lower()
    return set(item)

test = open("perfect_test.txt","r").read()
# test = re.sub(r"\n+","\n",test)
test_list = clean(test)
ends = ["lda", "tfidf", "infsnt"]

for end in ends:
    fold = "outs_" + end + "/"
    for i in range(10):
        fname = "out" + str(i) + ".txt"
        outf = open(fold + fname, "r").read()
        outlst = clean(outf)
        jname = "/jcrd" + str(i) + ".txt"
        jf = open("jaccard_" + end + jname, "w")
        for line in outlst:
            line_set = create_set(line)
            doc_set = create_set(test_list[i])
            jscore = nltk.jaccard_distance(line_set, doc_set)
            # print(jscore, set(word_tokenize(line)), set(word_tokenize(test_list[i])))
            jf.write(str(jscore) + "\n")

