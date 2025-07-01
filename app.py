import streamlit as st
import pandas as pd
import base64
from surprise import dump

# -----------------------------
# ✅ Page Config (Must be first)
# -----------------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")


# -----------------------------
# 📥 Load Data
# -----------------------------
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
data = pd.merge(ratings, movies, on="movieId")

# -----------------------------
# 📦 Load Trained Model
# -----------------------------
_, model = dump.load("svd_model")

# -----------------------------
# 🎯 Recommendation Function
# -----------------------------
def get_top_recommendations(user_id, n=5):
    all_titles = data['title'].unique()
    watched = data[data['userId'] == user_id]['title'].tolist()
    unseen = [title for title in all_titles if title not in watched]

    predictions = []
    for title in unseen:
        pred = model.predict(user_id, title)
        predictions.append((title, pred.est))

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return top_n, watched

# -----------------------------
# 🖥️ Streamlit UI
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("Get personalized movie suggestions based on your past ratings.")

# 📌 Select User
user_ids = sorted(data['userId'].unique())
selected_user = st.selectbox("👤 Select User ID", user_ids)

# 🎯 Show Recommendations
if st.button("🍿 Get Recommendations"):
    with st.spinner("Finding the best picks for you..."):
        recommended, watched = get_top_recommendations(selected_user)

    st.success(f"Top Picks for User {selected_user}")
    st.subheader("⭐ Recommended Movies:")
    for i, (title, rating) in enumerate(recommended, 1):
        st.write(f"**{i}. {title}** — Predicted Rating: ⭐ {round(rating, 2)}")

    st.markdown("---")
    with st.expander("📺 Movies You've Watched"):
        for title in watched[:10]:
            st.write(f"🔹 {title}")
