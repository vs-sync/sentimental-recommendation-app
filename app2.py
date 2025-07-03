# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:05:29 2025
@author: sayan
"""

import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import nltk
import string
import re
import os  # <-- add this
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer

# ========= Safe NLTK Downloads =========
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

def safe_nltk_download(pkg):
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg.split('/')[-1], download_dir=nltk_data_dir)

safe_nltk_download('corpora/stopwords')
safe_nltk_download('tokenizers/punkt')
safe_nltk_download('taggers/averaged_perceptron_tagger')
safe_nltk_download('corpora/wordnet')
safe_nltk_download('corpora/omw-1.4')

# ========= MODEL CLASS =========
class SentimentRecommenderModel:
    MODEL_NAME = 'stacking_model_compressed.joblib'
    VECTORIZER = 'tfidf-vectorizer.pkl'
    RECOMMENDER = 'user_final_rating2.joblib'
    CLEANED_DATA = 'cleaned_data.pkl'
    RAW_DATA = 'sample30.csv'

    def __init__(self):
        try:
            self.model = joblib.load(self.MODEL_NAME)
            self.vectorizer = pd.read_pickle(self.VECTORIZER)
            self.user_final_rating = joblib.load(self.RECOMMENDER)
            self.data = pd.read_csv(self.RAW_DATA)
            self.cleaned_data = pickle.load(open(self.CLEANED_DATA, 'rb'))
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            raise

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def getRecommendationByUser(self, user):
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

    def getSentimentRecommendations(self, user):
        if user in self.user_final_rating.index:
            recommendations = self.getRecommendationByUser(user)
            filtered_data = self.cleaned_data[self.cleaned_data.id.isin(recommendations)].copy()
            X = self.vectorizer.transform(filtered_data["reviews_full_text"].astype(str))
            filtered_data["predicted_sentiment"] = self.model.predict(X)

            temp = filtered_data[['id', 'predicted_sentiment']]
            temp_grouped = temp.groupby('id', as_index=False).count()
            temp_grouped.rename(columns={'predicted_sentiment': 'total_review_count'}, inplace=True)

            temp_grouped["pos_review_count"] = temp_grouped.id.apply(
                lambda x: temp[(temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"] / temp_grouped["total_review_count"] * 100, 2)

            sorted_products = temp_grouped.sort_values('pos_sentiment_percent', ascending=False)[0:5]
            return pd.merge(self.data, sorted_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])
        else:
            return None

    def classify_sentiment(self, review_text):
        review_text = self.preprocess_text(review_text)
        X = self.vectorizer.transform([review_text])
        return self.model.predict(X)

    def preprocess_text(self, text):
        text = text.lower().strip()
        text = re.sub(r"\[\s*\w*\s*\]", "", text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"\S*\d\S*", "", text)
        return self.lemma_text(text)

    def remove_stopword(self, text):
        return " ".join([word for word in text.split() if word.isalpha() and word not in self.stop_words])

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemma_text(self, text):
        word_pos_tags = nltk.pos_tag(word_tokenize(self.remove_stopword(text)))
        words = [self.lemmatizer.lemmatize(tag[0], self.get_wordnet_pos(tag[1])) for tag in word_pos_tags]
        return " ".join(words)

# ========= STREAMLIT APP =========
st.set_page_config(page_title="Sentiment-Based Recommender", layout="centered")
st.title("🛒 Sentiment-Based Product Recommender")
st.markdown("Get product recommendations based on sentiment analysis of reviews.")

# Initialize model (with error display if something breaks)
try:
    model = SentimentRecommenderModel()
except Exception:
    st.stop()

tab1, tab2 = st.tabs(["📌 Recommend by User", "📝 Classify Review"])

# --- Tab 1: Recommend Products ---
with tab1:
    user_input = st.text_input("Enter your User ID:")
    if st.button("Get Top 5 Recommendations"):
        if user_input:
            result = model.getSentimentRecommendations(user_input)
            if result is not None and not result.empty:
                st.success("Top 5 Recommended Products Based on Positive Sentiment:")
                st.dataframe(result.reset_index(drop=True))
            else:
                st.warning("User not found or no recommendations available.")
        else:
            st.warning("Please enter a valid user ID.")

# --- Tab 2: Classify Review ---
with tab2:
    review_text = st.text_area("Enter a product review:")
    if st.button("Predict Sentiment"):
        if review_text:
            prediction = model.classify_sentiment(review_text)
            if prediction[0] == 1:
                st.success("✅ Positive Review")
            else:
                st.error("❌ Negative Review")
        else:
            st.warning("Please enter a valid review.")
