import streamlit as st
from transformers import pipeline
import gc

st.header("Sentiment Analysis")
st.subheader("Enter your sentence below")

classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli')

text = st.text_area('Enter text here!')
candidate_labels = ['Positive', 'Neutral', 'Negative']

if text:
  out = classifier(text, candidate_labels)
  st.json(out)
  del out
  gc.collect()
