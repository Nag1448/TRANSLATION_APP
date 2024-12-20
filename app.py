from transformers import MarianMTModel, MarianTokenizer

def translate_text(text, source_lang, target_lang):
    # Load the tokenizer and model
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize and generate translation
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Example usage
source_language = "en"  # English
target_language = "fr"  # French
text = "Hello, how are you?"

translated_text = translate_text(text, source_language, target_language)
print("Translated Text:", translated_text)
