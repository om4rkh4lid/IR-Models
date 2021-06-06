import math
from numpy import dot
from numpy.linalg import norm

maxFiles = 3
alphabet = "ABCDEF"

queryVector = [1, 2, 3, 4]
weightMatrix = [
    [1,1,1,1],
    [2,2,2,2],
    [1,5,7,9],
]



# list(float) list(list(float)) -> dict{string:float}
def getVSModelSimilarity(queryVector, weightMatrix, count = maxFiles):
    raw_result = dict()
    for i in range(count):
        fileName = "D"+str(i+1)
        cosine_sim = dot(queryVector, weightMatrix[i])/(norm(queryVector)*norm(weightMatrix[i]))
        raw_result[fileName] = cosine_sim

    ordered_keys = sorted(raw_result, key=raw_result.get, reverse=True)
    ordered_result = dict()
    for k in ordered_keys:
        ordered_result[k] = raw_result[k]
    
    return ordered_result



if __name__ == "__main__":
    print(getVSModelSimilarity(queryVector, weightMatrix))