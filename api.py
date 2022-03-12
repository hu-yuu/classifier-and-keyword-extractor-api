from cgitb import text
from flask import Flask, request
import numpy as np
import requests
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import tfidfextractor
import json


import grpc
import zemberek_grpc.language_id_pb2 as z_langid
import zemberek_grpc.language_id_pb2_grpc as z_langid_g
import zemberek_grpc.normalization_pb2 as z_normalization
import zemberek_grpc.normalization_pb2_grpc as z_normalization_g
import zemberek_grpc.preprocess_pb2 as z_preprocess
import zemberek_grpc.preprocess_pb2_grpc as z_preprocess_g
import zemberek_grpc.morphology_pb2 as z_morphology
import zemberek_grpc.morphology_pb2_grpc as z_morphology_g
#java -jar zemberek-full.jar StartGrpcServer --dataRoot .\zemberek_data\
#java -jar zemberek-full.jar StartGrpcServer --dataRoot ./zemberek_data/
channel = grpc.insecure_channel('localhost:6789')
langid_stub = z_langid_g.LanguageIdServiceStub(channel)
normalization_stub = z_normalization_g.NormalizationServiceStub(channel)
preprocess_stub = z_preprocess_g.PreprocessingServiceStub(channel)
morphology_stub = z_morphology_g.MorphologyServiceStub(channel)


app = Flask(__name__)

svm = pickle.load(open('./pickles/LinearSVC.pickle', 'rb'))
vectorizer = pickle.load(open('./pickles/TfidfVectorizer.pickle', 'rb'))
myKwExtractor = tfidfextractor.TfIdfExtractor()

trstopwords =[]
f = open("./stopwords.txt","r", encoding='utf-8')
for i in f:
    i= re.sub('[^a-zA-ZÂâğüşöçıİĞÜŞÖÇ]', "", i)
    trstopwords.append(i)

def normalize(text):
    res = normalization_stub.Normalize(z_normalization.NormalizationRequest(input=text))
    if res.normalized_input:
        return res.normalized_input
    else:
        print('Problem normalizing input : ' + res.error)


def tokenize(text_arr):
    token_str = ""
    tokens = []
    
    text= re.sub('[^a-zA-ZÂâğüşöçıİĞÜŞÖÇ]', " ", text_arr)
    text = text.lower()
    tokens.append(text)
    
    return tokens

def stemming(text):
    stemmed = []
    stem_str = ""
    for token in text[0].split(" "):
        if token == "" :
            continue
        res = morphology_stub.AnalyzeSentence(z_morphology.SentenceAnalysisRequest(input=str(token)))
        if res.results[0].best.dictionaryItem.lemma.lower() != 'unk' and res.results[0].best.dictionaryItem.lemma.lower() not in trstopwords:
            stem_str += res.results[0].best.dictionaryItem.lemma.lower()+ " "
        elif res.results[0].best.dictionaryItem.lemma.lower() == 'unk' :
            stem_str += token+ " "
    stemmed.append(stem_str.strip())

    return stemmed[0]   


def onisleme(text):

    text = normalize(text)
    text = tokenize(text)
    text = stemming(text)
    return text






@app.route("/classify")
def classify():
    
    text = request.args.get('text')

    try:
        text = onisleme(text)
    except:
        return json.dumps(
            {
                "message": "error processing text"
            }
        , indent=4,), 500

    texts = [text, text]
    try:
        X = vectorizer.transform(texts)
    except:
        return json.dumps(
            {
                "message": "error vectorizing text"
            }, indent=4
        ), 500
    try:
        label = svm.predict(X[0])
        return json.dumps(
            {
                "label": label[0]
            }, indent=4, ensure_ascii=False
    ).encode('utf8')
    except:
        return json.dumps(
            {
                "message": "error classifying text"
            }
        ), 500

    
    return 
    
    
@app.route("/extract")
def extract():
    
    text = request.args.get('text')
    text = onisleme(text)

    try:
        ngram1 = myKwExtractor.extract(text, 10, 1)
        ngram2 = myKwExtractor.extract(text, 10, 2)
        return json.dumps({
            "N-Gram1": ngram1,
            "N-Gram2": ngram2
        }, indent=4, ensure_ascii=False).encode('utf8')

    except:
        return json.dumps({
            "message": "error extracting keywords"
        }, indent=4), 500





if __name__ == "__main__":
    app.run(debug=True, port=9998)
