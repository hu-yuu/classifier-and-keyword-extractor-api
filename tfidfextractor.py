import pickle
import pandas as pd


class TfIdfExtractor():
    def __init__(self):
        self.tfidfobj1 = pickle.load(open('./pickles/tfidfngram1.pickle', 'rb'))
        self.tfidfobj2 = pickle.load(open('./pickles/tfidfngram2.pickle', 'rb'))

    def __keyphrases(self, nmb, textname, a):
        rowelem = a.loc[textname, :]
        keyphraseList = dict(zip(list(a.columns), rowelem))


        topnmb = []
        for w in sorted(keyphraseList, key=keyphraseList.get, reverse=True):
            topnmb.append(w)
            if (len(topnmb) == nmb):
                break

        return topnmb

    def extract(self, text, keywordNum, ngram = 1):
                    
        txtlist = []
        txtlist.append("deneme")
        txtlist.append(text)

        if ngram == 1:
            vect = self.tfidfobj1.transform(txtlist)
            vectDF = pd.DataFrame(vect.toarray(), columns=self.tfidfobj1.get_feature_names())
        
        else:
            vect = self.tfidfobj2.transform(txtlist)
            vectDF = pd.DataFrame(vect.toarray(), columns=self.tfidfobj2.get_feature_names())


        try:
            returnval = self.__keyphrases(keywordNum, 1, vectDF)
            return returnval
        except:
            print("error extracting keywords")


