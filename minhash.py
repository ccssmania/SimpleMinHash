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

shingles = []
docs = []
matrix = {}
mult = 100
numDocs = 1
permutationNumber = 100
signatureMatrix = {}

dataFile = "data/dataSet_" + str(numDocs) + ".txt"

arrayIndex = []


#Functions
def sieve_of_eratosthenes(max_integer, start_):
    sieve = [True for _ in range(max_integer + 1)]
    sieve[0:1] = [False, False]
    for start in range(max_integer + 1):
        if sieve[start]:
            for i in range(2 * start, max_integer + 1, start):
                sieve[i] = False
    primes = []
    for i in range(start_, max_integer + 1):
        if sieve[i]:
            return i
    return primes

def hasFunction(a,b,m):
	per = {}
	for i in range(0,len(m)) :
		per.setdefault((a*i+b)%next_prime, i)
	return per



#Reading documents and creating input matrix
time_start = time.time()
print("*******************reading documents***********************")
with open(dataFile, "rU") as fp:  
	line = fp.readline().split(" ")
	cnt = 0
	shinCount = 0
	while cnt < numDocs * mult:
		docAux = [cnt,line[0]]
		docs.append(docAux) #saving the doc id
		del line[0]
		for index in range(0, len(line) - 2):
			shingle = line[index] + " " + line[index + 1] + " " + line[index + 2]
			#print(index, shingle, cnt)
			#time.sleep(0.1)
			if(shingle in shingles):
				
				ind = shingles.index(shingle)
				matrix[ind].setdefault(cnt, 1)
			else:
				shingles.append(shingle)
				matrix.setdefault(shinCount, {})
				matrix[shinCount].setdefault(cnt,1)
				arrayIndex.append(shinCount)
				shinCount += 1

			
		line = fp.readline().split(" ")
		cnt += 1


#permutations
next_prime = sieve_of_eratosthenes(len(shingles)*2,len(shingles)) #Next Prime
print("Time ", time.time()-time_start, " Seconds")
print("***********************************************************")
print("*****************Shingle Universe**************************")

print(len(arrayIndex))
print("***********************************************************")

time_start = time.time()
print("******************making a permutations********************")
for i in range(0, permutationNumber):
	arrayIndex = hasFunction(random.randint(1,101),random.randint(1,101), list(matrix.keys()))
	for j in range(0, numDocs*mult):
		signatureMatrix.setdefault(j,[])
		for key, value in arrayIndex.iteritems():
			if(matrix[value].get(j)):
				signatureMatrix[j].append(key)
				break


resultMatrix = []
treshold = 0.5

print("Time ", time.time()-time_start, " Seconds")
print("***********************************************************")

print("*****************Calculing Results*************************")
time_start = time.time()
for i in range(0, numDocs*mult):
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

