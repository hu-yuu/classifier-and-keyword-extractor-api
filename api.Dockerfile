FROM python:3.8.13

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN mkdir pickles
COPY ./pickles ./pickles

COPY api.py classifier.py preprocessing.py tfidfextractor.py stopwords.txt ./

CMD [ "python", "./api.py" ]