import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/HP/Desktop/AB_NYC_2019.csv")

    # Drop unnecessary columns
    df = df.drop(columns=["id", "name", "host_name", "last_review", "neighbourhood"])
    
    # Handle missing values
    df.fillna(0, inplace=True)

    return df

@st.cache_resource
def perform_clustering(df, n_clusters=5):
    features = df[[
        "latitude", "longitude", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count", "availability_365",
        "accommodates", "bedrooms", "bathrooms"
    ]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)

    df["cluster"] = cluster_labels
    return df, kmeans, scaler

def predict_cluster(scaler, kmeans, user_input_df):
    scaled_input = scaler.transform(user_input_df)
    cluster = kmeans.predict(scaled_input)[0]
    return cluster

# Main Streamlit App
def main():
    st.set_page_config(page_title="NYC Airbnb Clustering & Price Estimator", layout="centered")
    st.title("🏘️ Airbnb Cluster Explorer & Price Estimator (Unsupervised)")

    df = load_data()
    df, kmeans, scaler = perform_clustering(df)

    st.subheader("📍 Customize your preferences:")
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=40.75)
        longitude = st.number_input("Longitude", value=-73.98)
        accommodates = st.slider("Accommodates", 1, 16, 2)
        bedrooms = st.slider("Bedrooms", 0, 5, 1)
        bathrooms = st.slider("Bathrooms", 0, 4, 1)
    with col2:
        min_nights = st.slider("Minimum Nights", 1, 30, 3)
        num_reviews = st.slider("Number of Reviews", 0, 500, 50)
        reviews_per_month = st.slider("Reviews per Month", 0.0, 10.0, 1.2)
        listings_count = st.slider("Host Listing Count", 1, 50, 2)
        availability = st.slider("Availability (days/year)", 0, 365, 180)

    user_input = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "minimum_nights": min_nights,
        "number_of_reviews": num_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": listings_count,
        "availability_365": availability,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms
    }])

    if st.button("🔍 Find My Cluster & Predict Price"):
        cluster = predict_cluster(scaler, kmeans, user_input)
        st.success(f"📊 Your input belongs to *Cluster #{cluster}*")

        avg_price = df[df["cluster"] == cluster]["price"].mean()
        st.info(f"💰 Based on similar listings, the *estimated average price* is: *${avg_price:.2f}*")

        st.subheader("📌 Similar Listings in Your Cluster:")
        st.dataframe(df[df["cluster"] == cluster][[
            "latitude", "longitude", "price", "accommodates", "bedrooms", "bathrooms"
        ]].head(10))

if _name_ == "_main_":
    main()