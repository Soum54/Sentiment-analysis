import streamlit as st
import pandas as pd
from transformers import pipeline
import requests

# Load the local sentiment analysis pipeline
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}  # Replace with your actual token

# Function to query the Hugging Face API
def query_huggingface_api(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.json()

# Define a function to analyze sentiment locally with the pipeline
def analyze_sentiment(text):
    result = pipe(text)
    sentiment = result[0]['label']

    # Map the sentiment to an emoji
    if sentiment == "POSITIVE":
        emoji = "😊"
    elif sentiment == "NEGATIVE":
        emoji = "😢"
    elif sentiment == "NEUTRAL":
        emoji = "😐"
    else:
        emoji = "🤔"  # Fallback for any other cases

    return f"{sentiment.capitalize()} {emoji}"

# Streamlit UI elements
st.title("Sentiment Analysis using Hugging Face")

st.write("""
Upload a CSV file with a 'Text' column or input your own text to perform sentiment analysis.
You can choose to analyze sentiment locally using a pre-trained model or via the Hugging Face API.
""")

# User text input for single sentiment analysis
st.header("Analyze Sentiment of a Single Text")
user_input = st.text_input("Enter text for sentiment analysis")

# Selection between local model and Hugging Face API
analysis_method = st.radio(
    "Choose sentiment analysis method:",
    ('Local Model', 'Hugging Face API')
)

# Add analyze button for single text input
if st.button("Analyze Text"):
    if user_input:
        if analysis_method == 'Local Model':
            sentiment_result = analyze_sentiment(user_input)
        else:
            api_response = query_huggingface_api(user_input)
            sentiment_result = api_response[0]['generated_text'] if 'generated_text' in api_response[0] else "No sentiment returned"
        st.write(f"Sentiment: {sentiment_result}")
    else:
        st.warning("Please enter some text to analyze.")

# File upload for batch sentiment analysis
st.header("Batch Sentiment Analysis via CSV")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Add analyze button for CSV file
if uploaded_file:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Check if the 'Text' column exists
    if 'Text' in df.columns:
        st.write("Data Preview:")
        st.write(df.head())

        if st.button("Analyze CSV"):
            # Apply sentiment analysis to each row in the DataFrame
            st.write("Performing sentiment analysis...")
            if analysis_method == 'Local Model':
                df['Sentiment'] = df['Text'].apply(analyze_sentiment)
            else:
                df['Sentiment'] = df['Text'].apply(lambda x: query_huggingface_api(x)[0]['generated_text'] if 'generated_text' in query_huggingface_api(x)[0] else "No sentiment returned")

            # Show the result in Streamlit
            st.write("Sentiment analysis results:")
            st.write(df.head())

            # Provide a download button for the updated CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name='sentiment_analysis_results_with_emoji.csv',
                mime='text/csv',
            )
    else:
        st.error("The uploaded CSV does not contain a 'Text' column.")
else:
    st.write("Please upload a CSV file to perform batch sentiment analysis.")
