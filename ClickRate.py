

from pyspark import SparkContext
from pyspark.mllib.linalg import SparseVector
import numpy as np
from pyspark.mllib.regression import LabeledPoint

dacsample = "/home/dusan/Documents/dac_sample.txt"  # Should be some file on your system
sc = SparkContext("local", "Simple App")

dacsample = open(dacsample)
dacData = [unicode(x.replace('\n', '').replace('\t', ',')) for x in dacsample]
rawData  = (sc
            .parallelize(dacData, 1)  # Create an RDD
            .zipWithIndex()  # Enumerate lines
            .map(lambda (v, i): (i, v))  # Use line index as key
            .partitionBy(2, lambda i: not (i < 50026))  # Match sc.textFile partitioning
            .map(lambda (i, v): v))  # Remove index



print "---------------------------"
print rawData.take(1)
print "---------------------------"

weights = [.8, .1, .1]
seed = 42
# Use randomSplit with weights and seed
rawTrainData, rawValidationData, rawTestData = rawData.randomSplit([.8 ,.1, .1], 42)
# Cache the data
rawTrainData.cache()
rawValidationData.cache()
rawTestData.cache()


def createOneHotDict(inputData):
   
    return inputData.flatMap(lambda x : x).distinct().zipWithIndex().collectAsMap()



def parsePoint(point):
    
    y = map(lambda x : (x), point.split(","))
    y.pop(0)
    z1 = map(lambda x : (y.index(x), x), y)
    z2 =[(i,x) for i,x in enumerate(y)]
    return z2

parsedTrainFeat = rawTrainData.map(parsePoint)


print "---------------------------"
print parsedTrainFeat.take(1)
print "---------------------------"

def oneHotEncoding(rawFeats, OHEDict, numOHEFeats):
   
    
    xx = filter(lambda item : ctrOHEDict.has_key(item) , rawFeats)
           
    sparseVector = SparseVector(numOHEFeats, sorted(map(lambda x : OHEDict[x], sorted(xx))), np.ones(len(xx)))
    
    return sparseVector


def parseOHEPoint(point, OHEDict, numOHEFeats):
   
    x = parsePoint(point)
    sparseVector = oneHotEncoding(x, OHEDict, numOHEFeats)
    
    return LabeledPoint(point[0], sparseVector)

ctrOHEDict = createOneHotDict(parsedTrainFeat)
numCtrOHEFeats = len(ctrOHEDict.keys())

print rawTrainData.take(1)
OHETrainData = rawTrainData.map(lambda point: parseOHEPoint(point, ctrOHEDict, numCtrOHEFeats))
OHETrainData.cache()




OHEValidationData = rawValidationData.map(lambda point: parseOHEPoint(point, ctrOHEDict, numCtrOHEFeats))
OHEValidationData.cache()

from pyspark.mllib.classification import LogisticRegressionWithSGD

# fixed hyperparameters
numIters = 50
stepSize = 10.
regParam = 1e-6
regType = 'l2'
includeIntercept = True




model0 = LogisticRegressionWithSGD.train(data=OHETrainData, iterations=numIters, step=stepSize,regParam=regParam, regType=regType, intercept=includeIntercept)
sortedWeights = sorted(model0.weights)

sc.stop()


