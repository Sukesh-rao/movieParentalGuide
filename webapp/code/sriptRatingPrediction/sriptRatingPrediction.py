### Package import
#from ssl import _PeerCertRetDictType
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import collections
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model


lemmatizer = WordNetLemmatizer()

### Userdefined CSS
with open("movieCSS.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

### Adding Background Image
import base64
def addBackground(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
addBackground('appback.png')   

### Module for text cleaning
def scriptClean(script):
    ### Lower case conversion
    cleanScript = script.lower()
    
    ### Removing special characters
    cleanScript = [re.sub('[^A-Za-z ]+', '', str(i)) for i in cleanScript.split(' ')]
    
    ### Excluding stopwords
    cleanScript = [i for i in cleanScript if i not in stopwords.words('english')]
    
    ### Root word extraction
    cleanScript = [lemmatizer.lemmatize(i) for i in cleanScript]
    
    ### Excluding words with less than or equal to length 2
    cleanScript = [lemmatizer.lemmatize(i) for i in cleanScript if len(i) > 2]
    
    return ' '.join(cleanScript)


### Application Header
st.markdown('<div style = "background-color: rgb(194, 136, 43); text-align: center;"> <h5 style = "color: white; padding: 5px;"> Movie parental guide predictive profiling from raw movie script using AI  </h5> </div>', unsafe_allow_html=True)
st.markdown('''<br>''', unsafe_allow_html=True)
st.markdown('''<h6 style="color: white"> Enter Movie Script: </h6>''', unsafe_allow_html=True)

### Text Input
txt = st.text_area('Text to analyze', '''
    Please paste script here!
    ''', label_visibility = 'collapsed')
#st.button("Get Parential Guide Rating")

### Recommendations
st.markdown('''<h6 style="color: white"> Script Analysis Report: </h6>''', unsafe_allow_html=True)

st.markdown('''<div style="background-color: white">''', unsafe_allow_html=True)
## Adding Expanders for Report Display
with st.expander("Score Report", expanded = True):
    ## Defining layout
    col1, col2 = st.columns(2)

    ### Mapping dummy codes to corresponding rating
    def mappingFun(cd):
        if cd == 0:
            return 'mild'
        elif cd == 1:
            return 'severe'
        elif cd == 2:
            return 'none'
        else:
            return 'moderate'

    ## Data Preparation for prediction
    # Word tokenization using keras
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #tk.texts_to_matrix(texts, mode='tfidf')
    tokenized_texts = tokenizer.texts_to_sequences([txt])
    X = pad_sequences(tokenized_texts, maxlen=3000)

    modelFileList = ['intense_2023-04-10_10_14_34_171450.h5', 'profanity_2023-04-10_10_10_45_589936.h5', 'sex_2023-04-10_10_09_20_703872.h5', 'violence_2023-04-10_10_02_53_256942.h5']
    modelPredList = []

    with st.spinner('Generating prediction from Deep Models. Please wait...'):
        
        for mdl in modelFileList:
            model = load_model(mdl)
            prediction =  [np.argmax(i) for i in model.predict(X)]
            finalPred = mappingFun(prediction[0])
            modelPredList.append(finalPred)
 

    ## Cards for ratings
    col1.markdown('''<h6> Violence Rating:</h6>''', unsafe_allow_html=True)
    img = Image.open('%s.png'%(modelPredList[3]))
    col1.image(img)

    col2.markdown('''<h6> Intense Rating:</h6>''', unsafe_allow_html=True)
    img = Image.open('%s.png'%(modelPredList[0]))
    col2.image(img)

    col1.markdown('''<h6> Sex Rating:</h6>''', unsafe_allow_html=True)
    img = Image.open('%s.png'%(modelPredList[2]))
    col1.image(img)

    col2.markdown('''<h6> Profanity Rating:</h6>''', unsafe_allow_html=True)
    img = Image.open('%s.png'%(modelPredList[1]))
    col2.image(img)

with st.expander("Script Summary", expanded = False):
    col1, col2 = st.columns(2)

    ### Word cloud
    col1.markdown('''<h6> Word Cloud:</h6>''', unsafe_allow_html=True)
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='rgb(255, 253, 208)',
                    stopwords = set(STOPWORDS),
                    min_font_size = 10).generate(scriptClean(txt))
    wordcloud.to_file('wordCloud.png')
    col1.image(Image.open('wordCloud.png'))

    ### Frequent Bigram Table
    counter=collections.Counter(nltk.bigrams(scriptClean(txt).split(' ')))

    ### Bigram df
    w1 = []
    w2 = []
    value = [] 
    for i,j in zip(list(counter), list(counter.values())) :
        w1.append(i[0])
        w2.append(i[1])
        value.append(j)
    bigramDf = pd.DataFrame({'word1': w1, 'word2': w2, 'freq': value}).sort_values(['freq'], ascending = False)
    bigramDf = bigramDf.reset_index(drop=True)

    col2.markdown('''<h6> Frequent Bi-Gram (Top 10):</h6>''', unsafe_allow_html=True)
    col2.table(bigramDf.head(10))


st.markdown('''</div>''', unsafe_allow_html=True)