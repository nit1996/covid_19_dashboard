import streamlit as st
import nltk
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pickle
from wordcloud import WordCloud,STOPWORDS
nltk.download('stopwords') 
import matplotlib.pyplot as plt
import spacy
nltk.download('wordnet')
nlp = spacy.blank("en")
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()
st.set_option('deprecation.showPyplotGlobalUse', False)
html_temp = """ 
    <div style ="background-color:tomato;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit ML App</h1> 
    </div> 
    """
st.markdown(html_temp, unsafe_allow_html = True) 
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return text
try:   
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    loaded_model = pickle.load(open('classification.pkl',"rb",))
    gb_loaded_model =pickle.load(open('gb_classification.pkl',"rb"))
except UnicodeDecodeError:
    pass

def predict_category(text):
    text = vectorizer.transform([text])
    text = text.toarray()
    pred = loaded_model.predict(text)
    return pred

def pred_cat(text):
    text = vectorizer.transform([text])
    text = text.toarray()
    pred = gb_loaded_model.predict(text)
    return pred 

def cloud(text,min_font,max_font,bg_color,max_word):
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color=bg_color,max_words=max_word,
                   stopwords=stopwords, max_font_size=max_font, min_font_size=min_font)
    wc.generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()
    
def main():
    options = ['Prediction','NLP','WordCloud']
    choice = st.sidebar.selectbox('Choose one Actvity',options)
    if choice=='NLP':
        html_temp1 = """ 
        
        <h2 style ="text-align:center;font-family:Apple Chancery, cursive;font-size:36px;">Natural Language Processing</h2> 
        
         """
        st.markdown(html_temp1, unsafe_allow_html = True) 
        st.markdown('<style>body{background-color:#FFFFF0;}</style>',unsafe_allow_html=True)
        if st.button('Click to See Definitions'):
            st.info('Tokenization :The process of tokenizing or splitting a string, text into a list of tokens. One can think of token as                       parts like a word is a token in a sentence, and a sentence is a token in a paragraph. ')
            st.info('Stopwords: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been                                 programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search                          query.')
            st.info('Stemming: Reduces the corpus of words the model is exposed to and explicitly correlates words with similar meanings.')
            st.info('Lemmatizing : Process of grouping together the inflected forms of a word so they can be analyzed as a single                                term,identified by the word lemma')
        text = st.text_input('Enter Text','please type here')
        choice = ['Remove Punctuation','Tokenization','Removing StopWords','Stemming','Lemmatizing','Cleaned Text']
        options = st.sidebar.selectbox('Select choice', choice)
        if st.button('Generate Result'):
            if options == 'Remove Punctuation':
                result = "".join([char for char in text if char not in string.punctuation])
                st.text('Original text::\n{}'.format(text))
                st.text('After Punctuation Removal::\n{}'.format(result))
            elif options == 'Tokenization':
                
                tokens = re.split('\W+',text)
                st.text('Original text::\n{}'.format(text))
                st.text('After tokenization::\n{}'.format(tokens))
                
            elif options == 'Removing StopWords' :
                no_stopwords = " ".join([word for word in text.lower().split() if word not in stopwords])
                st.text('Original text::\n{}'.format(text))
                st.text('stopwords are::\n{}'.format(stopwords))
                st.text('After Stopwords Removal::\n{}'.format(no_stopwords))
              
            elif options == 'Stemming':
                stemmend_text = ps.stem(text)
                st.text('Original text::\n{}'.format(text))
                st.text('Text After Stemming::\n{}'.format(stemmend_text))
            
            elif options == 'Lemmatizing':
                lemmatized_text = wn.lemmatize(text)
                st.text('Original text::\n{}'.format(text))
                st.text('Text After Lemmatizing::\n{}'.format(lemmatized_text))
            else:
                result = clean_text(text)
                st.text('Original text::\n{}'.format(text))
                st.text('After cleaning::\n{}'.format(result))
            
    if choice=='Prediction':
        st.markdown('<style>body{background-color: oldLace;}</style>',unsafe_allow_html=True)
        st.title('Email Spam Detection App')
        st.markdown("""
        <style>
        .big-font {
        font-size:30px !important;
        background-color:#aaf0d1;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="big-font">Predict an Email message as ham or spam</p>', unsafe_allow_html=True)
        
        text_message = st.text_area('Enter E-mail message','please enter...')
        models = ['Random Forest','Gradient Boost']
        choose_model = st.sidebar.selectbox('Choose one model',models)
        labels = ['Spam','Ham']
        if st.button('Generate Prediction'):
            st.text('Email message::\n{}'.format(text_message))
            if choose_model =='Random Forest':
                result = predict_category(text_message)
                st.write('Entered message is')
                st.success(result)
            elif choose_model =='Gradient Boost':
                result = pred_cat(text_message)
                st.success('Entered message is :{}'.format(result))
            else:
                st.error('Please select any model')
    if choice=='WordCloud':
        html_temp2 = """ 
        
        <h2 style ="text-align:center;font-family:Apple Chancery, cursive;font-size:40px;">WordCloud</h2> 
        
         """
        st.markdown(html_temp2, unsafe_allow_html = True) 
        st.markdown('<style>body{background-color:#FAF0E6;}</style>',unsafe_allow_html=True)
        st.markdown("""
        <style>
        .big-font {
        font-size:23px !important;
        font-weight:bold;
        background-color:#DB7093;
        color:white;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button('Click to see Info'):
            st.markdown('<p class="big-font">Word Cloud is a data visualization technique used for representing text data in which the size                         of each word indicates its frequency or importance. Significant textual data points can be highlighted using a word                         cloud. </p>', unsafe_allow_html=True)
        max_word = st.sidebar.slider("Max words", 200, 3000, 200)
        min_font = st.sidebar.slider("Min Font Size", 10, 100, 40)
        max_font = st.sidebar.slider("Max Font Size", 50, 350, 60)
        bg_color = st.sidebar.radio('choose background',('white','black'))
        text_message = st.text_area('Enter E-mail message','please enter...')
        if st.button('Generate WordCloud'):
            cloud(text_message,min_font,max_font,bg_color,max_word)


if __name__ == '__main__':
    main()
