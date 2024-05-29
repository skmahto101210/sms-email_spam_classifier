import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('punkt')

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def tranform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))
            
    return " ".join(y)

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    tranformed_sms = tranform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([tranformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # st.text(result)
    # 4. Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
