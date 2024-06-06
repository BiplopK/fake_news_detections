import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

vectorization = TfidfVectorizer()

vector_form = pickle.load(open('G:\\machinlearning\\fake-news-detection\\notebooks\\vector.pkl', 'rb'))
load_model = pickle.load(open('G:\\machinlearning\\fake-news-detection\\notebooks\\model.pkl', 'rb'))

def process_news(content):
    news=re.sub('[^a-zA-z]',' ',content)
    news=content.lower()
    # Removing Punctuation
    news = news.translate(str.maketrans('', '', string.punctuation))
    # Tokenizing Words
    tokens = word_tokenize(news)
    # Removing Stopwords and Lemmatization
    wordLemm = WordNetLemmatizer()
    final_news = [wordLemm.lemmatize(w) for w in tokens if w not in stopwords.words('english')]
    return " ".join(final_news)

def fake_news(news):
    news=process_news(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction



if __name__ == '__main__':
    st.title('Fake News Classification app ')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "",height=200)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        if prediction_class == [1]:
            st.success('Real News')
        if prediction_class == [0]:
            st.warning('Fake News')