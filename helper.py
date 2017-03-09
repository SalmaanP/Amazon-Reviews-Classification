import numpy as np
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


def loadData(trainingFile, testingFile, debug=False):
    """Take in training and testing files and split into reviews and labels"""
    
    trainFile = open(trainingFile, 'r').read().splitlines()
    testFile = open(testingFile, 'r').read().splitlines()
    train_labels = [x.split("\t", 1)[0] for x in trainFile]
    train_reviews = [x.split("\t", 1)[1] for x in trainFile]
    if debug:
        test_reviews = [x.split("\t", 1)[1] for x in testFile]
        test_labels = [x.split("\t", 1)[0] for x in testFile]
        return train_reviews, test_reviews, train_labels, test_labels
    else:
        return train_reviews, testFile, train_labels
    

    
def preProcess(train_reviews):
    stemmer = nltk.stem.porter.PorterStemmer()
    stopwords = open('stopwords_en.txt').read()
    for index, review in enumerate(train_reviews):
        sentence = review.lower().translate(None, string.punctuation)
        acc = ''
        for word in sentence.split():
            if word not in stopwords and len(word) > 3:
                acc+=str(stemmer.stem(word)) + ' '
        train_reviews[index] = acc
        
    return train_reviews


def createMatrices(train_data, test_data):
    """Takes in processed training and testing data, outputs respective sparse matrices with TF-IDF values"""
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    train_matrix = vectorizer.fit_transform(train_data)
    test_matrix = vectorizer.transform(test_data)
    return train_matrix, test_matrix


def findSimilarities(train_data, test_data):
    """Takes in entire training and testing data (sparse matrices), outputs similarities as a list of numpy arrays"""
    
    similarities = []
    for index, vector in enumerate(test_data):
        cosine = cosine_similarity(vector, train_data)[0]
        similarities.append(cosine)
    return similarities


def findKNearest(similarity_vector, k):
    """Takes in one similarity vector and number of neighbors to find, Returns K Nearest Neighbors indices"""
    
    return np.argsort(-similarity_vector)[:k]
     

def predict(nearestNeighbors, labels):
    """Takes in a list of K nearestNeighbors, and training labels, outputs 1 or -1"""

    positive = 0
    negative = 0
    for neighbor in nearestNeighbors:
        if int(labels[neighbor]) == 1:
            positive+=1
        else:
            negative+=1
    if positive > negative:
        return 1
    else:
        return -1

    





