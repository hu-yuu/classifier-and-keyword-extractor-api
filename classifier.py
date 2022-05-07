import pickle


class Classifier:
    def __init__(self):
        self.svm = pickle.load(open('./pickles/LinearSVC.pickle', 'rb'))
        self.vectorizer = pickle.load(open('./pickles/TfidfVectorizer.pickle', 'rb'))

    def vectorize(self, text):
        texts = [text, text]
        vectors = self.vectorizer.transform(texts)

        return vectors

    def classify(self, vectors):
        label = self.svm.predict(vectors[0])

        return label