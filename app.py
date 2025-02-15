import subprocess
import streamlit as st


# Function to install missing libraries
def install_libraries():
    try:
        import matplotlib
    except ImportError:
        st.warning("🔴 Installing matplotlib...")
        subprocess.run(["pip", "install", "matplotlib"])

    try:
        import nltk
    except ImportError:
        st.warning("🔴 Installing nltk...")
        subprocess.run(["pip", "install", "nltk"])


# Run installation
install_libraries()


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER Lexicon (Only required once)
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to apply sentiment analysis with a confidence score
def get_sentiment_vader(text):
    text = str(text).strip()  # Ensure input is a string
    if len(text) < 5:  # Ignore very short text
        return "Neutral", 0.0  # Default to Neutral with 0 confidence

    score = sia.polarity_scores(text)["compound"]

    # Convert score to sentiment labels
    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("📊 AI-Based Public Sentiment Analysis")
st.write("Analyze public opinion on government policies using AI.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file containing public text data", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Ensure the dataset has valid text-based columns (exclude Date, numeric fields)
    text_columns = [col for col in df.columns if df[col].dtype == 'object' and col.lower() not in ["date", "id"]]

    # Check if there are valid text columns available
    if not text_columns:
        st.error("⚠️ No valid text columns found! Please upload a dataset with Title or Content fields.")
    else:
        st.subheader("📌 Select Text Column for Analysis")
        text_column = st.selectbox("Select the text column:", text_columns)

        # **Ensure "Text" column is created**
        if "Title" in df.columns and "Content" in df.columns:
            df["Text"] = df["Title"].fillna('') + " " + df["Content"].fillna('')
        else:
            df["Text"] = df[text_column].fillna("")

        # **Verify "Text" column exists before running sentiment analysis**
        if "Text" not in df.columns or df["Text"].isnull().all():
            st.error("⚠️ No valid text data found! Ensure Title or Content is selected.")
        else:
            # **Apply Sentiment Analysis**
            with st.spinner("Analyzing sentiment..."):
                df[['Sentiment', 'Confidence Score']] = df['Text'].apply(lambda x: pd.Series(get_sentiment_vader(x)))

            # Remove rows with "Error" in Sentiment
            df_sentiment_cleaned = df[df["Sentiment"] != "Error"]

            # Check if valid data is available
            if df_sentiment_cleaned.empty:
                st.error("⚠️ All rows had errors in sentiment analysis. Please upload a dataset with valid text.")
            else:
                # Count sentiment distribution
                sentiment_counts = df_sentiment_cleaned['Sentiment'].value_counts()

                # Show cleaned data in a table
                st.subheader("🔹 Processed Data Sample")
                st.dataframe(df_sentiment_cleaned[['Text', 'Sentiment', 'Confidence Score']])

                # **📌 Display Each Row in Detail**
                st.subheader("🔎 View Each Row's Detailed Information")
                for index, row in df_sentiment_cleaned.iterrows():
                    with st.expander(f"🔹 Row {index+1} - Sentiment: {row['Sentiment']} (Confidence: {row['Confidence Score']:.2f})"):
                        st.write(f"**Text:** {row['Text']}")
                        st.write(f"**Predicted Sentiment:** {row['Sentiment']} (Confidence Score: {row['Confidence Score']:.2f})")

                # **📊 Matplotlib Bar Chart**
                st.subheader("📊 Sentiment Distribution")

                # Create a Matplotlib figure
                plt.figure(figsize=(8, 5))
                plt.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "gray", "red"])

                # Customize plot
                plt.title("Sentiment Analysis Distribution")
                plt.xlabel("Sentiment")
                plt.ylabel("Number of Posts")

                # Show the Matplotlib chart in Streamlit
                st.pyplot(plt)

                # **Download Processed Data**
                st.download_button(label="📥 Download Processed Data",
                                   data=df_sentiment_cleaned.to_csv(index=False).encode("utf-8"),
                                   file_name="sentiment_analysis_results.csv",
                                   mime="text/csv")

st.write("Developed by: Deng | AI Developer 🚀")
