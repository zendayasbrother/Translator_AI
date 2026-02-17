import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect

# Load model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Mapping dictionary for NLLB specific language codes
LANG_MAP = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "auto": "eng_Latn" 
}

def detect_language(text):
    try:
        if not text or not any(char.isalpha() for char in text):
            return "en"
        return detect(text)
    except Exception:
        return "en" 

def translate(text, source='auto', target='en'):
    # FIXED: Added a default fallback to "eng_Latn" if the code is not in the map
    src_lang_code = LANG_MAP.get(source, "eng_Latn") 
    tgt_lang_code = LANG_MAP.get(target, "eng_Latn") 

    # Prepare the tokenizer for the specific source language
    tokenizer.src_lang = src_lang_code
    
    # We pass src_lang here as well to ensure proper encoding
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get the ID for the target language token
    # NLLB uses the target language code as the first forced token (BOS)
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
    
    translated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tgt_lang_id, 
        max_length=150
    )
    
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

print("\n" + "=" * 40)
enquiry_text = input("Enter your text: ").strip()
target_pref = input("Enter desired target language code (e.g., 'en', 'fr', 'es'): ").strip().lower()
print("=" * 40)

# Detect source language
detect_lang = detect_language(enquiry_text)

# Perform translation
translated_enq = translate(enquiry_text, source=detect_lang, target=target_pref)

print(f"\nOriginal Enquiry: {enquiry_text}")
print(f"Detected Language: {detect_lang}")

# Final Response output
print(f"Final Response ({target_pref}): {translated_enq}")
