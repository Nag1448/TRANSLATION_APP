import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

st.title("Translation App")

text = st.text_input("Enter text to translate:")
source_lang = st.selectbox("Source Language", ["en", "fr", "de"])
target_lang = st.selectbox("Target Language", ["fr", "en", "te"])

if st.button("Translate"):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    st.write("Translated Text:", translated_text)
