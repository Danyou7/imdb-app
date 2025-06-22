import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# === Load model dan tokenizer ===
@st.cache_resource
def load_components():
    model = load_model("model1.keras") 
    with open("tokenizer.pkl", "rb") as f: 
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_components()

# === Preprocessing teks ===
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-ZÀ-ú\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# === Fungsi prediksi ===
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded, verbose=0)[0][0]
    label = "Positif" if prediction >= 0.5 else "Negatif"
    return prediction, label

# === Visualisasi ===
def plot_pie(counts):
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#66bb6a', '#ef5350'])
    ax.axis('equal')
    return fig

def plot_common_words(df):
    pos_words = " ".join(df[df['label'] == "Positif"]['clean_text']).split()
    neg_words = " ".join(df[df['label'] == "Negatif"]['clean_text']).split()

    top_pos = Counter(pos_words).most_common(10)
    top_neg = Counter(neg_words).most_common(10)

    pos_df = pd.DataFrame(top_pos, columns=['word', 'count'])
    neg_df = pd.DataFrame(top_neg, columns=['word', 'count'])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x='count', y='word', data=pos_df, ax=axs[0], color='green')
    axs[0].set_title("Top Words (Positive)")
    sns.barplot(x='count', y='word', data=neg_df, ax=axs[1], color='red')
    axs[1].set_title("Top Words (Negative)")
    st.pyplot(fig)

def plot_ngrams_by_sentiment(df, top_n=10):
    sentiments = ['Positif', 'Negatif']
    ngram_titles = ['Unigram', 'Bigram', 'Trigram']
    ngram_ranges = [(1,1), (2,2), (3,3)]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for row_idx, (ngram_range, title) in enumerate(zip(ngram_ranges, ngram_titles)):
        for col_idx, sentiment in enumerate(sentiments):
            text_series = df[df['label'] == sentiment]['clean_text']
            vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
            X = vectorizer.fit_transform(text_series)
            sum_words = X.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
            ngram_df = pd.DataFrame(words_freq, columns=['Ngram', 'Count'])

            sns.barplot(x='Count', y='Ngram', data=ngram_df, ax=axes[row_idx, col_idx], color='green' if sentiment == 'Positif' else 'red')
            axes[row_idx, col_idx].set_title(f"{title} - {sentiment}")
            axes[row_idx, col_idx].set_xlabel("")

    plt.tight_layout()
    st.pyplot(fig)

# === Analisis CSV ===
def analyze_csv(df, text_column):
    df['clean_text'] = df[text_column].astype(str).apply(preprocess_text)
    sequences = tokenizer.texts_to_sequences(df['clean_text'].tolist())
    padded = pad_sequences(sequences, maxlen=100)
    probs = model.predict(padded, verbose=0).flatten()
    labels = np.where(probs >= 0.5, "Positif", "Negatif")

    df['prob'] = probs
    df['label'] = labels

    with st.expander("📊 Distribusi Sentimen"):
        st.pyplot(plot_pie(df['label'].value_counts()))
    with st.expander("📌 Kata Umum"):
        plot_common_words(df)
    with st.expander("📊 N-Gram Positif vs Negatif"):
        plot_ngrams_by_sentiment(df)

    st.subheader("🧾 Tabel Hasil Analisis")
    st.dataframe(df[[text_column, 'label', 'prob']])
    st.download_button("📥 Unduh CSV", df.to_csv(index=False), file_name="hasil_sentimen.csv")

# === UI Layout ===
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .css-1d391kg {padding-top: 2rem;}
        h1 {color: #1f77b4;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard Analisis Sentimen")

# === Layout Sidebar ===
with st.sidebar:
    st.markdown("## 🧠 Analisis Sentimen")
    menu = st.radio("Pilih Analisis", ["✍️ Input Manual", "📁 Analisis CSV", "ℹ️ Tentang"])

# === Tampilan Utama Berdasarkan Pilihan ===
if menu == "✍️ Input Manual":
    st.subheader("✍️ Input Teks")
    user_input = st.text_area("Masukkan teks ulasan", height=150)
    if st.button("Prediksi Sentimen"):
        if user_input.strip() == "":
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            clean_input = preprocess_text(user_input)
            prob, label = predict_sentiment(clean_input)
            st.success(f"**Label Sentimen:** {label}")
            st.info(f"**Probabilitas Positif:** {prob:.4f}")

elif menu == "📁 Analisis CSV":
    st.subheader("📁 Upload File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV dengan kolom teks", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        text_columns = df.select_dtypes(include='object').columns.tolist()
        if text_columns:
            selected_column = st.selectbox("Pilih kolom teks:", text_columns)
            if st.button("🔍 Analisis"):
                analyze_csv(df, selected_column)
        else:
            st.error("Tidak ditemukan kolom teks dalam file.")

elif menu == "ℹ️ Tentang":
    st.markdown("""
    ### Tentang Aplikasi
    Dev : Danu Tirta
                
    NPM : 10123285
                
    Aplikasi ini digunakan untuk:
    - Memprediksi sentimen (positif/negatif) dari teks.
    - Menampilkan visualisasi analisis CSV (pie chart, kata umum, n-gram).
    - Mendukung input manual & file CSV.  
    """)
