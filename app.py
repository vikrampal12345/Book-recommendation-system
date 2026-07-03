import streamlit as st
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

@st.cache_resource
def load_files():
    with open("book.pkl", "rb") as f:
        book = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vector.pkl", "rb") as f:
        vector = pickle.load(f)

    return book, model, vector

book, model, vector = load_files()

def recommend(book_ob):
    index1 = book[book['Book-Title'] == book_ob].index[0]
    distances, indices = model.kneighbors(vector[index1:index1+1])

    recommended_books = []

    for i in indices[0][1:]:
        recommended_books.append({
            "title": book.iloc[i]["Book-Title"],
            "image": book.iloc[i]["Image-URL-L"]
        })

    return recommended_books

st.title("📚 Book Recommendation System")


with st.form("recommend_form"):
    book_title = st.selectbox(
        "Select a Book",
        sorted(book["Book-Title"].unique()),
        index=None,
        placeholder="🔍 Search here..."
    )

    submitted = st.form_submit_button("Recommend")

if submitted:
    if book_title is None:
        st.warning("Please select a book first.")
    else:
        recommendation = recommend(book_title)

        cols = st.columns(5)

        for col, item in zip(cols, recommendation):
            with col:
                st.image(item["image"], width=150)
                st.caption(item["title"])