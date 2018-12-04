from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
spark = SparkSession.builder.appName('lsh_spark').getOrCreate()


from tqdm import tqdm
import numpy as np
import pprint
import time
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords


shingles = dict()
docs = []
matrix = {}
mult = 100
numDocs = 25
permutationNumber = 100
signatureMatrix = {}

dataFile = "data/dataSet_" + str(numDocs) + ".txt"

arrayIndex = []
stopWords = set(stopwords.words('english'))

time_total = time.time()
data = []


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

sh_count = 0
with open(dataFile, "rU") as fp:  
	lines = fp.read().split("\nt")
	cnt = 0
	
	for line in tqdm(lines):
		line = line.split(" ")
		del line[0]
		matrix.setdefault(cnt,dict())
		for index in range(0, len(line) - 2):
			if(line[index] in stopWords):
				shingle = line[index] + " " + line[index + 1] + " " + line[index + 2]
				if shingles.get(shingle) or shingles.get(shingle) == 0:
					matrix[cnt].setdefault(shingle,shingles.get(shingle))
				else:
					shingles.setdefault(shingle,sh_count)
					matrix[cnt].setdefault(shingle,sh_count)
					sh_count += 1

				
		
		line = fp.readline().split(" ")
		cnt += 1
size = len(list(shingles))
cnt = 0
for key,value in tqdm(matrix.items()):
	aux = []
	for index, sh in value.items():
		aux.append(sh)
	data.append((key,Vectors.sparse(size,sorted(list(aux)),np.ones(len(list(aux))))))
next_prime = sieve_of_eratosthenes(size*2,size)
sc = spark.sparkContext
distData = sc.parallelize(data)

#df = spark.createDataFrame(data, ["id", "features"])
df = spark.createDataFrame(distData, ["id", "features"])

key = Vectors.dense([1.0, 0.0])

mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5,  seed=next_prime)
model = mh.fit(df)
dft = model.transform(df)
model.approxSimilarityJoin(dft,dft, 0.6, distCol="JaccardDistance").select(
			col("datasetA.id").alias("idA"),
			col("datasetB.id").alias("idB"),
			col("JaccardDistance")).filter("idA != idB").show()

