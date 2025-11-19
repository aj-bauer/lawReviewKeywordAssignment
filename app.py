from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
# import pandas as pd
import joblib
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import skops.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.preprocessing import remove_stopwords
import nltk
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import csv

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), "model", "label_maker.skops")
unknown_types = sio.get_untrusted_types(file=model_path)
model = sio.load(model_path, trusted=unknown_types)

# Define output keywords used in the model
subj_terms = ['Administrative Law',
               'Business Organizations Law',
               'Civil Law and Procedure',
               'Civil Rights and Discrimination',
               'Commercial Law',
               'Communications Law',
               'Constitutional Law',
               'Courts',
               'Criminal Law and Procedure',
               'Domestic Law',
               'Education Law',
               'Environmental Law',
               'Health Law and Policy',
               'Housing Law',
               'Immigration Law',
               'Insurance Law',
               'Intellectual Property Law',
               'International Law',
               'Jurisdiction',
               'Jurisprudence',
               'Law and Society',
               'Legal Profession',
               'Legislation',
               'National Security Law',
               'Natural Resources Law',
               'Science and Technology Law',
               'Supreme Court of the United States',
               'Taxation',
               'Torts',
               'Trade Regulation']

# Load our docs and train our vectorizer
doc_path = os.path.join(os.path.dirname(__file__), "data", "docs.csv")
docs = []
with open(doc_path, "r") as csvfile:
    reader_variable = csv.reader(csvfile, delimiter=",")
    next(reader_variable) # skip header
    for row in reader_variable:
        docs = docs + row
# docs = pd.read_csv(doc_path)
vectorizer = TfidfVectorizer(max_features=2500, max_df=0.9).fit(docs)

# define input text preprocessor
def doc_preprep_tfidf(abstract):
  doc = str(abstract)
  # Remove stopwords
  doc = remove_stopwords(doc)
  # Split the documents into tokens.
  tokenizer = RegexpTokenizer(r'\w+')
  # Convert to lowercase.
  doc = doc.lower()  
  # Split into words.
  doc = tokenizer.tokenize(doc) 
  # Remove numbers, but not words that contain numbers.
  doc = [token for token in doc if not token.isnumeric()]
  # Remove words that are only one or two characters.
  doc = [token for token in doc if len(token) > 2]
  # Lemmatize the documents.
  lemmatizer = WordNetLemmatizer()
  doc = [lemmatizer.lemmatize(token) for token in doc]
  # Remove other stop words
  stop_words = ['comment', 'note', 'article', 'argues']
  doc = [token for token in doc if token not in stop_words]
  # Concat our tokens back into a string
  doc = " ".join(doc)
  return doc

# Initialize FastAPI app
app = FastAPI()

# Allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mesa8414-label-maker.onrender.com"],  # In production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body schema
class TextInput(BaseModel):
    abstract: str

# Define output
class ModelOutput(BaseModel):
  prediction: str

# Mount the 'frontend' directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve index.html at root
@app.get("/")
def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# Define the prediction endpoint
@app.post("/predict")
async def predict(input: TextInput):
    # Generate TFIDF
    data = doc_preprep_tfidf(input.abstract)
    tfidf = vectorizer.transform(data)
    
    # Preduct output using the model
    prediction = model.predict(tfidf)

    # Transform into list of keywords
    df = pd.DataFrame(prediction, columns=col_names)
    cols = df.columns[(df == 1).any()].tolist()
    prediction_string = ", ".join(cols)
  
    # reutrn data
    return ModelOutput(prediction = prediction_string)
