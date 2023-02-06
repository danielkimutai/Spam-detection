import pandas
import numpy
import pickle
import streamlit as st
import contractions
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.collocations import *
import string
from nltk.stem import WordNetLemmatizer
import re


#preprocessing

#lowercasing the texts
def lowercase_text(text):
        """This function converts characters to lowercase"""
        return text.lower()
    

#removing punctuation in the texts
def remove_punct(text):
    text= "".join([c for c in text if c not in string.punctuation])
    return text

def lemmat(text):
    lemma=WordNetLemmatizer()
    tokens=re.split('\W+', text)
    text=' '.join([lemma.lemmatize(token)for token in tokens])
    return text

stop_words=set(stopwords.words('english'))
def remove_stop(x):
    return " ".join( [word for word in str(x).split() if word not in stop_words])


def remove_small_words(text):
        """This function removes words with length 1 or 2"""
        clean = re.compile(r'\b\w{1,2}\b')
        
        return re.sub(clean, '', text)
def expand_contractions(text):
        """Expand shortened words, e.g. don't to do not"""
        text = contractions.fix(text)
        
        return text

def preprocess(text):
    """This function preprocesses text"""
    text = remove_stop(text)
    text = remove_small_words(text)
    text = lemmat(text)
    text = remove_punct(text)
    text = lowercase_text(text)
    text=expand_contractions(text)
    return text

vectorizer=pickle.load(open('vectorizerr.sav','rb'))
model=pickle.load(open('grid_model.sav','rb'))

st.title("Email/Spam Classifier")
st.write("A model that classifies whether an email is a spam/ham")

input_sms = st.text_input("Enter the email")

if st.button("Classify"):
    processed_sms=preprocess(input_sms)
    vector_input=vectorizer.transform([processed_sms])
    input_array=vector_input.toarray()
    prediction=model.predict(input_array)[0]
    if prediction == 0:
        st.success("This is a Ham")
    else:
        st.warning("Sparm Alert !!!!!")