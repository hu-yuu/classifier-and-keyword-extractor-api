
import re
import numpy as np

import grpc
import zemberek_grpc.language_id_pb2_grpc as z_langid_g
import zemberek_grpc.normalization_pb2 as z_normalization
import zemberek_grpc.normalization_pb2_grpc as z_normalization_g
import zemberek_grpc.preprocess_pb2_grpc as z_preprocess_g
import zemberek_grpc.morphology_pb2 as z_morphology
import zemberek_grpc.morphology_pb2_grpc as z_morphology_g

#java -jar zemberek-full.jar StartGrpcServer --dataRoot .\zemberek_data\
#java -jar zemberek-full.jar StartGrpcServer --dataRoot ./zemberek_data/

channel = grpc.insecure_channel('zemberek:6789')
langid_stub = z_langid_g.LanguageIdServiceStub(channel)
normalization_stub = z_normalization_g.NormalizationServiceStub(channel)
preprocess_stub = z_preprocess_g.PreprocessingServiceStub(channel)
morphology_stub = z_morphology_g.MorphologyServiceStub(channel)


class Preprocessor:
    def __init__(self):
        self.trstopwords =[]
        f = open("./stopwords.txt","r", encoding='utf-8')
        for i in f:
            i= re.sub('[^a-zA-ZÂâğüşöçıİĞÜŞÖÇ]', "", i)
            self.trstopwords.append(i)

    def __normalize(self, text):
        res = normalization_stub.Normalize(z_normalization.NormalizationRequest(input=text))
        if res.normalized_input:
            return res.normalized_input
        else:
            print('Problem normalizing input : ' + res.error)


    def __tokenize(self, text_arr):
        token_str = ""
        tokens = []
        
        text= re.sub('[^a-zA-ZÂâğüşöçıİĞÜŞÖÇ]', " ", text_arr)
        text = text.lower()
        tokens.append(text)
        
        return tokens

    def __stemming(self, text):
        stemmed = []
        stem_str = ""
        for token in text[0].split(" "):
            if token == "" :
                continue
            res = morphology_stub.AnalyzeSentence(z_morphology.SentenceAnalysisRequest(input=str(token)))
            if res.results[0].best.dictionaryItem.lemma.lower() != 'unk' and res.results[0].best.dictionaryItem.lemma.lower() not in self.trstopwords:
                stem_str += res.results[0].best.dictionaryItem.lemma.lower()+ " "
            elif res.results[0].best.dictionaryItem.lemma.lower() == 'unk' :
                stem_str += token+ " "
        stemmed.append(stem_str.strip())

        return stemmed[0]   


    def process(self, text):

        text = self.__normalize(text)
        text = self.__tokenize(text)
        text = self.__stemming(text)
        
        return text