from flask import Flask, request
from flask_restful import Resource, Api
import tfidfextractor, classifier, preprocessing


app = Flask(__name__)
api = Api(app)

preprocessor = preprocessing.Preprocessor()
kwExtractor = tfidfextractor.TfIdfExtractor()
txtClassify = classifier.Classifier()



class KeywordsRoute(Resource):
    def get(self):
        text = request.args.get('text')
        if (text is None or text == "" ):
            return {
            "message": "text param is required"
        }, 400

        outboundVars = extract(text)
        return outboundVars[0], outboundVars[1]


class LabelRoute(Resource):
    def get(self):
        text = request.args.get('text')
        if (text is None or text == "" ):
            return {
            "message": "text param is required"
        }, 400

        outboundVars = classify(text)       
        return outboundVars[0], outboundVars[1]

api.add_resource(KeywordsRoute, '/keywords')
api.add_resource(LabelRoute, '/label')



def classify(text):

    global httpStatus
    global outboundPayload
        
    if (len(text.split()) < 10):
        return {
            "message": "Input text must be longer than 10 words"
        }, 400

    try:
        text = preprocessor.process(text)
    except:
        return {
                "message": "error processing text"
            } , 400

    
    try:
        X = txtClassify.vectorize(text)
    except:
        return {
                "message": "error vectorizing text"
            }, 400
    try:
        label = txtClassify.classify(X)
        return {
                "label": label[0]
            }, 200
    except:
        return {
                "message": "error classifying text"
            }, 400
 
    
    
def extract(text):

    text = preprocessor.process(text)

    if (len(text.split()) < 10):
        return {
            "message": "Input text must be longer than 10 words"
        }, 400

    try:
        ngram1 = kwExtractor.extract(text, 10, 1)
        ngram2 = kwExtractor.extract(text, 10, 2)
        return {
            "N-Gram1": ngram1,
            "N-Gram2": ngram2
        }, 200

    except:
        return {
            "message": "error extracting keywords"
        }, 400


if __name__ == "__main__":
    app.run(debug=True, port=9998)
