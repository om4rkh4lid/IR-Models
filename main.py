import os
import math
from flask import *
import random
from numpy import dot
from numpy.linalg import norm

app = Flask(__name__)


alphabet = "ABCDEF"
maxFiles = 10

def checkFilesExist():
    """ Checks if files exist and return True if they do, False o.w.
    """
    return os.path.exists("D1")

def getRandomString(characters = alphabet):
    """ Returns a random string of 1-100 characters from A-F.
    """   
    length = random.randint(1,10)
    rstring = ""
    for _ in range(length):
        rstring += random.choice(characters)
    return rstring

def createFiles(count = maxFiles):
    """ Creates (count) random files labeled D1 through D(count),
        each contains a random string using getRandomString().
    """
    for i in range(1, count + 1):
        filename = "D" + str(i)
        f = open(filename, "w+")
        f.write(getRandomString())
    global maxFiles
    if(count != maxFiles):
        maxFiles = count


def getStringWeights(s):
    """ Returns a list of weights for each character in (s),
        where weight = number of occurrences of letter / length of (s).
    """ 
    length = len(s)
    weightList = list()
    for letter in alphabet:
        w = s.count(letter) / length
        weightList.append(w)
    return weightList
    
def getFileWeights():
    """ Returns a dictionary {filename:weight} for all files created,
        where weight is a list of string weights provided by getStringWeights().
    """
    weights = dict()
    for i in range(1, maxFiles + 1):
        filename = "D"+str(i)
        fstring = open(filename).read()
        fw = getStringWeights(fstring)
        weights[filename] = fw
    return weights

def sortSimilarities(similarities, descending= True):
    """ Sort a dictionary based on value.
    """
    ordered_keys = sorted(similarities, key=similarities.get, reverse=descending)
    ordered_result = dict()
    for k in ordered_keys:
        ordered_result[k] = similarities[k]
    return ordered_result

def getQueryWeightsAsList(values):
    """ Converts the dictionary produced from the query into a vector
        of weights that corresponds to the vector of all possible characters
        in the files (alphabet).
    """
    weights = list()
    for letter in alphabet:
        if(letter in values):
            weights.append(values[letter])
        else:
            weights.append(0)
    return weights


def getQueryWeights(query):
    """ Cleans and converts the query to a dictionary {letter:weight}
        and returns a list of query weights using getQueryWeightsAsList(),
        where the weights are provided in the query.
    """
    query = query.replace("<", "")
    query = query.replace(">", "")
    queryValues = dict()
    params = query.split(";")
    for p in params:
        key = p[0]
        value = float(p[2:])
        queryValues[key] = value
    return getQueryWeightsAsList(queryValues)


def calculateStatModelSimilarity(query, data):
    similarities = dict()
    for i in range(1, maxFiles + 1):
        filename = "D" + str(i)
        fileweights = data[filename]
        similarities[filename] = sum([a*b for a,b in zip(query,fileweights)])
    return similarities

#################################################################################

# void -> dict{string:float}
def getCharacterIDFs(count = maxFiles, characters = alphabet):
    """ Returns a list containing the Inverse Document Frequency for
        each character.
        idf(i) = log2(number of files / no. of docs term appears in)
    """
    idfs = dict()
    docs = dict()

    for letter in characters:
        docs[letter] = 0

    for i in range(1, count + 1):
        filename = "D"+str(i)
        fstring = open(filename).read()
        for letter in characters:
            if letter in fstring:
                docs[letter] += 1
    print(docs)
    for letter in characters:
        idfs[letter] = math.log(count/docs[letter])

    return list(idfs.values())

# string -> int
def getMostFrequentChar(string, chars = alphabet):
    """ Returns the count of the most frequent character in a given string.
    """
    max = 0
    # iterate over chars in alphabet because it would be redundant to iterate
    # over each letter in the string.
    for letter in chars:
        count = string.count(letter)
        if count > max:
            max = count
    return max

# void -> list(list(float))
def getFileTFMatrix(count = maxFiles, chars = alphabet):
    """ Returns a list of vectors (matrix) of term frequencies
        for each character in each file.
        term frequency = freq. of char in file / most frequent char in file.
    """
    matrix = list()  
    for i in range(1, count + 1):
        filename = "D"+str(i)
        vector = list()
        fstring = open(filename).read()
        mostFrequent = getMostFrequentChar(fstring)
        for letter in chars:
            vector.append(fstring.count(letter)/mostFrequent)
        matrix.append(vector)
    
    return matrix


# list(float) list(list(float)) -> list(list(float))
def getWeightMatrix(idfs, frequencyMatrix):
    """ returns matrix of tfidf for each character in each file
        tfidf(ij) = text_frequency(ij) * inverse_doc_freq(i)
    """
    matrix = list()
    for vector in frequencyMatrix:
        tf_idf = list()
        for a, b in zip(vector, idfs):
            tf_idf.append(a*b)
        matrix.append(tf_idf)
    return matrix

# dict{string:float} -> list(float)
def getQueryVector(query, chars = alphabet):
    """ gets the Term Frequency vector for the query
    """
    vector = list()
    mostFrequent = getMostFrequentChar(query)
    for letter in chars:
        vector.append(query.count(letter)/mostFrequent)
    return vector

# list(float) list(list(float)) -> dict{string:float}
def getVSModelSimilarity(queryVector, weightMatrix, count = maxFiles):
    """ calculates the cosine similarity between the query vector and each tf-idf vector
        in the weights matrix, orders the results in descending order, then returns a
        dictionary of names and similiarities.
    """
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



def calculateVSModelSimilarity(query):
    # vector of inverse document frequencies
    idfs = getCharacterIDFs()
    print(idfs)
    # matrix of term frequncies for each term in each document
    fileTFMatrix = getFileTFMatrix()
    # matrix of weight(ij) = tf(ij) * idf(i)
    weightMatrix = getWeightMatrix(idfs, fileTFMatrix)
    # term frequency vector for the query
    queryWeights = getQueryVector(query)
    # cosine similarity with query vector for each vector in the matrix
    similarities = getVSModelSimilarity(queryWeights, weightMatrix)
    return similarities

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/stat_model', methods = ['POST', 'GET'])
def statModelSearchResult():
    weights = getFileWeights()
    query = request.form['query']   
    queryWeights = getQueryWeights(query)
    similarities = calculateStatModelSimilarity(queryWeights, weights)
    similarities = sortSimilarities(similarities)
    return render_template("results.html", result = similarities, modelUsed = "Statistical Model", numDocs = maxFiles, mostSimilar = list(similarities)[0], sim = list(similarities.values())[0])

@app.route('/vs_model', methods = ['POST', 'GET'])
def VSModelSearchResult():
    query = request.form['query']   
    # queryWeights = getQueryWeights(query)
    similarities = calculateVSModelSimilarity(query)
    similarities = sortSimilarities(similarities)
    return render_template("results.html", result = similarities, modelUsed = "Vector Space Model", numDocs = maxFiles, mostSimilar = list(similarities)[0], sim = list(similarities.values())[0])

@app.route('/generate_data', methods = ['GET'])
def generateData():
    createFiles()
    result = checkFilesExist()
    return jsonify({"result": result})

@app.route('/check_files', methods = ['GET'])
def checkFiles():
    # code to check if files exist
    result = checkFilesExist()
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run()
