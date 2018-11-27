from __future__ import division
import os
import re
import random
import time
import binascii
from bisect import bisect_right
from heapq import heappop, heappush
import itertools
import numpy as np
import sys
import random
from sklearn.metrics import jaccard_similarity_score
import pprint
import time
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from tqdm import tqdm



shingles = []
docs = []
matrix = {}
mult = 100
numDocs = 10
permutationNumber = 100
signatureMatrix = {}

dataFile = "data/dataSet_" + str(numDocs) + ".txt"

arrayIndex = []
stopWords = set(stopwords.words('english'))

time_total = time.time()


def jaccard_similarity(x, y):
	intersection_cardinality = len(set(x).intersection(set(y)))
	union_cardinality = len(set(x).union(set(y)))
	return intersection_cardinality / float(union_cardinality)
#Functions
def sieve_of_eratosthenes(max_integer, start_):
    sieve = [True for _ in range(max_integer + 1)]
    sieve[0:1] = [False, False]
    for start in range(max_integer + 1):
        if sieve[start]:
            for i in range(2 * start, max_integer + 1, start):
                sieve[i] = False
    for i in range(start_, max_integer + 1):
        if sieve[i]:
            return i
    return False
def hasFunction(a,b,m):
	per = {}
	for i in range(0,len(m)) :
		per.setdefault((a*i+b)%next_prime, m[i])
	return per

def binarySearch(l,item,init,end):
	if(len(l) <= 0 or (end-1 == init and l[init] != item)):
		return False
	#print("end ", end , "init ", init, " list ",l[init],"item",item, "l", l)
	if(init >= end): 
		return False
	aux = l[init]
	if((init == len(l) - 1 and item != aux) or  (init == 0 and item != aux)):
		return False
	if(aux == item):
		return init
	if(item < aux):
		#print("first")
		return binarySearch(l,item,int(init/2),init)
	else:
		#print("second")
		return binarySearch(l,item,int((init+end) / 2),end)



#Reading documents and creating input matrix
time_start = time.time()
print("*******************reading documents***********************")
with open(dataFile, "rU") as fp:  
	lines = fp.read().split("\nt")
	cnt = 0
	for line in tqdm(lines):
		line = line.split(" ")
		del line[0]
		for index in range(0, len(line) - 2):
			if(line[index] in stopWords):
				shingle = line[index] + " " + line[index + 1] + " " + line[index + 2]
				#print(index, shingle, cnt)
				#time.sleep(0.1)
				if matrix.get(shingle):
					matrix[shingle].setdefault(cnt,1)
				else:
					matrix.setdefault(shingle, {cnt:1})

			
		line = fp.readline().split(" ")
		cnt += 1

#pprint.pprint(matrix)
#permutations
next_prime = sieve_of_eratosthenes(len(matrix)*2,len(matrix)) #Next Prime
print("Time ", time.time()-time_start, " Seconds")
print("***********************************************************")
print("*****************Shingle Universe**************************")

print(len(matrix))
print("***********************************************************")

time_start = time.time()
print("******************making permutations********************")
for i in tqdm(range(0, permutationNumber)):
	arrayIndex = hasFunction(random.randint(1,101),random.randint(1,101), list(matrix))
	for j in range(0, numDocs*mult):
		signatureMatrix.setdefault(j,[])
		for key, value in arrayIndex.iteritems():
			if(matrix[value].get(j)):
				signatureMatrix[j].append(key)
				break

#pprint.pprint(signatureMatrix)
resultMatrix = []
treshold = 0.5

print("Time ", time.time()-time_start, " Seconds")
print("***********************************************************")

print("*****************Calculing Results*************************")
time_start = time.time()
for i in tqdm(range(0, numDocs*mult)):
	#aux = []
	for j in range(i+1, numDocs*mult):
		js = jaccard_similarity_score(signatureMatrix[i], signatureMatrix[j])
		if(js>= treshold and i != j) :
			resultMatrix.append("docs : " + str(i) + " -------> " + str(j) + "         *similarity  " + str(js));
print("Time ", time.time()-time_start, " Seconds")
print("***********************************************************")
print("****results with a treshold equal or greater than 0.5******")
pprint.pprint(resultMatrix)
print("***********************************************************")

print("Total Time Execution : ", (time.time() - time_total) /60, " min")
print("Total of docs with a similarity equal or greater than ", treshold, " : " + str(len(resultMatrix)))