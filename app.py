# app.py - Flask server for neural machine translation
from flask import Flask, render_template, request, jsonify
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import time
import os
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Enhanced language code mappings
LANGUAGE_CODES = {
    # Existing languages
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'pt': 'Portuguese',
    'ja': 'Japanese',
    'ko': 'Korean',
    
    # Added Indian regional languages
    'bn': 'Bengali',
    'te': 'Telugu',
    'ta': 'Tamil',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'ur': 'Urdu',
    'or': 'Odia',
    'as': 'Assamese',
    
    # Added widely spoken global languages
    'id': 'Indonesian',
    'ms': 'Malay',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'sw': 'Swahili',
    'fa': 'Persian',
    'pl': 'Polish',
    'nl': 'Dutch',
    'it': 'Italian',
    'uk': 'Ukrainian',
    'el': 'Greek',
    'he': 'Hebrew',
    'hu': 'Hungarian',
    'cs': 'Czech',
    'sv': 'Swedish',
    'ro': 'Romanian',
    'fi': 'Finnish'
}

# List of available language pairs from Helsinki-NLP
# This is not exhaustive but covers many common languages
AVAILABLE_DIRECT_PAIRS = {
    'en-hi', 'hi-en', 'en-ta', 'ta-en', 'en-te', 'te-en', 'en-mr', 'mr-en',
    'en-bn', 'bn-en', 'en-ml', 'ml-en', 'en-ur', 'ur-en', 'en-gu', 'gu-en',
    'en-pa', 'pa-en', 'en-or', 'or-en', 'en-fr', 'fr-en', 'en-de', 'de-en',
    'en-es', 'es-en', 'en-ru', 'ru-en', 'en-zh', 'zh-en', 'en-ar', 'ar-en',
    'en-pt', 'pt-en', 'en-ja', 'ja-en', 'en-ko', 'ko-en', 'en-it', 'it-en',
    'en-nl', 'nl-en', 'en-pl', 'pl-en', 'en-tr', 'tr-en', 'en-cs', 'cs-en',
    'en-fi', 'fi-en', 'en-sv', 'sv-en', 'en-ro', 'ro-en', 'en-hu', 'hu-en',
    'en-el', 'el-en', 'en-he', 'he-en', 'en-th', 'th-en', 'en-vi', 'vi-en',
    'en-id', 'id-en', 'fr-de', 'de-fr', 'fr-es', 'es-fr', 'de-es', 'es-de'
}

# Model configuration
model_cache = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m2m_model = None
m2m_tokenizer = None

def get_model_name(source, target):
    """Get the appropriate Hugging Face model name for the language pair"""
    if source == 'en' and target != 'en':
        return f'Helsinki-NLP/opus-mt-en-{target}'
    elif source != 'en' and target == 'en':
        return f'Helsinki-NLP/opus-mt-{source}-en'
    else:
        # For non-English to non-English, we can try to find direct models
        return f'Helsinki-NLP/opus-mt-{source}-{target}'

def load_model(source, target):
    """Load or retrieve translation model from cache with improved fallback strategies"""
    model_key = f"{source}-{target}"
    if model_key in model_cache:
        return model_cache[model_key]
    
    # First try: Direct model if available
    try:
        if f"{source}-{target}" in AVAILABLE_DIRECT_PAIRS:
            model_name = get_model_name(source, target)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(device)
            model_cache[model_key] = (model, tokenizer)
            print(f"Loaded model {model_name}")
            return model, tokenizer
    except Exception as e:
        print(f"Direct model loading failed for {source}-{target}: {e}")
    
    # Second try: English pivot translation for languages without direct models
    if source != 'en' and target != 'en':
        try:
            # Check if source-to-English and English-to-target models exist
            if f"{source}-en" in AVAILABLE_DIRECT_PAIRS and f"en-{target}" in AVAILABLE_DIRECT_PAIRS:
                print(f"Using English pivot for {source}-{target}")
                source_en_model = get_model_name(source, 'en')
                en_target_model = get_model_name('en', target)
                
                source_tokenizer = MarianTokenizer.from_pretrained(source_en_model)
                source_model = MarianMTModel.from_pretrained(source_en_model).to(device)
                
                target_tokenizer = MarianTokenizer.from_pretrained(en_target_model)
                target_model = MarianMTModel.from_pretrained(en_target_model).to(device)
                
                model_cache[model_key] = ('pivot', (source_model, source_tokenizer), (target_model, target_tokenizer))
                return model_cache[model_key]
        except Exception as e:
            print(f"English pivot translation failed for {source}-{target}: {e}")
    
    # Third try: Use M2M100 multilingual model as fallback
    global m2m_model, m2m_tokenizer
    try:
        if m2m_model is not None and m2m_tokenizer is not None:
            print(f"Using M2M100 fallback for {source}-{target}")
            model_cache[model_key] = ('m2m100', m2m_model, m2m_tokenizer)
            return model_cache[model_key]
        else:
            print(f"Loading M2M100 model for {source}-{target}")
            m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(device)
            model_cache[model_key] = ('m2m100', m2m_model, m2m_tokenizer)
            return model_cache[model_key]
    except Exception as e:
        print(f"M2M100 fallback failed for {source}-{target}: {e}")
    
    # If all attempts fail
    print(f"No translation model available for {source}-{target}")
    return None

def translate_text(text, source_lang, target_lang, model_size='balanced'):
    """Enhanced translate text function with fallback model handling"""
    start_time = time.time()
    
    # Load model for the specific language pair
    model_data = load_model(source_lang, target_lang)
    
    if not model_data:
        return {
            "error": f"Translation model for {source_lang} to {target_lang} not available",
            "suggestion": "Try using English as an intermediate language"
        }, 404
    
    try:
        # Direct translation with MarianMT
        if isinstance(model_data, tuple) and len(model_data) == 2:
            model, tokenizer = model_data
            num_beams = 4 if model_size == 'balanced' else (2 if model_size == 'fast' else 6)
                
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                translated = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=num_beams,
                    early_stopping=True
                )
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # Pivot translation through English
        elif model_data[0] == 'pivot':
            _, (source_model, source_tokenizer), (target_model, target_tokenizer) = model_data
            
            # Translate source to English
            inputs = source_tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                translated_en = source_model.generate(**inputs, max_length=512, num_beams=4)
            en_text = source_tokenizer.decode(translated_en[0], skip_special_tokens=True)
            
            # Translate English to target
            inputs = target_tokenizer(en_text, return_tensors="pt").to(device)
            with torch.no_grad():
                translated = target_model.generate(**inputs, max_length=512, num_beams=4)
            translated_text = target_tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # M2M100 multilingual model fallback
        elif model_data[0] == 'm2m100':
            _, model, tokenizer = model_data
            
            # Set source language
            tokenizer.src_lang = source_lang
            
            # Encode input text
            encoded = tokenizer(text, return_tensors="pt").to(device)
            
            # Generate translation with appropriate beam search
            num_beams = 4 if model_size == 'balanced' else (2 if model_size == 'fast' else 6)
            with torch.no_grad():
                generated_tokens = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                    num_beams=num_beams,
                    max_length=512
                )
            
            # Decode the generated tokens
            translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        elapsed_time = time.time() - start_time
        return {
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "processing_time": elapsed_time,
            "model_size": model_size,
            "model_type": model_data[0] if isinstance(model_data, tuple) and len(model_data) > 0 and isinstance(model_data[0], str) else "direct"
        }
    
    except Exception as e:
        print(f"Translation error: {e}")
        return {"error": str(e), "suggestion": "Please try another language pair"}, 500

def detect_language(text):
    """Enhanced language detection based on character sets"""
    # Indian language detection
    if bool(re.search(r'[\u0900-\u097F]', text)):
        return 'hi'  # Hindi
    if bool(re.search(r'[\u0980-\u09FF]', text)):
        return 'bn'  # Bengali
    if bool(re.search(r'[\u0B00-\u0B7F]', text)):
        return 'or'  # Odia
    if bool(re.search(r'[\u0B80-\u0BFF]', text)):
        return 'ta'  # Tamil
    if bool(re.search(r'[\u0C00-\u0C7F]', text)):
        return 'te'  # Telugu
    if bool(re.search(r'[\u0C80-\u0CFF]', text)):
        return 'kn'  # Kannada
    if bool(re.search(r'[\u0D00-\u0D7F]', text)):
        return 'ml'  # Malayalam
    if bool(re.search(r'[\u0A80-\u0AFF]', text)):
        return 'gu'  # Gujarati
    if bool(re.search(r'[\u0A00-\u0A7F]', text)):
        return 'pa'  # Punjabi
    if bool(re.search(r'[\u0600-\u06FF]', text)) and bool(re.search(r'[\u0627\u0644\u0633\u0644\u0627\u0645]', text)):
        return 'ur'  # Urdu (Arabic script with specific Urdu patterns)
    if bool(re.search(r'[\u0980-\u09FF]', text)) and bool(re.search(r'[\u09BE\u09CC]', text)):
        return 'as'  # Assamese (similar to Bengali but with specific characters)
    if bool(re.search(r'[\u0900-\u097F]', text)) and bool(re.search(r'[\u0915\u094D\u0937]', text)):
        return 'mr'  # Marathi (Devanagari with specific conjuncts)
    
    # Asian languages
    if bool(re.search(r'[\u4e00-\u9FFF]', text)):
        return 'zh'  # Chinese
    if bool(re.search(r'[\u3040-\u30ff\u3400-\u4DBF]', text)):
        return 'ja'  # Japanese
    if bool(re.search(r'[\uAC00-\uD7A3]', text)):
        return 'ko'  # Korean
    if bool(re.search(r'[\u0E00-\u0E7F]', text)):
        return 'th'  # Thai
    if bool(re.search(r'[\u0370-\u03FF]', text)):
        return 'el'  # Greek
    
    # Middle Eastern
    if bool(re.search(r'[\u0600-\u06FF]', text)):
        return 'ar'  # Arabic
    if bool(re.search(r'[\u0590-\u05FF]', text)):
        return 'he'  # Hebrew
    if bool(re.search(r'[\u0600-\u06FF\u FB8A]', text)) and bool(re.search(r'[\u067E\u0686\u06AF]', text)):
        return 'fa'  # Persian (Farsi)
    
    # Cyrillic script
    if bool(re.search(r'[а-яА-Я]', text)):
        # Try to distinguish between Russian and Ukrainian
        if bool(re.search(r'[іїєґ]', text)):
            return 'uk'  # Ukrainian (has specific letters)
        return 'ru'  # Russian
    
    # Latin script with special characters
    if bool(re.search(r'[áéíóúüñ¿¡]', text)):
        return 'es'  # Spanish
    if bool(re.search(r'[àâçéèêëîïôùûüÿœæ]', text)):
        return 'fr'  # French
    if bool(re.search(r'[äöüßẞ]', text)):
        return 'de'  # German
    if bool(re.search(r'[ąćęłńóśźż]', text)):
        return 'pl'  # Polish
    if bool(re.search(r'[åæø]', text)):
        return 'sv'  # Swedish
    if bool(re.search(r'[ăîșț]', text)):
        return 'ro'  # Romanian
    if bool(re.search(r'[áéíóúőű]', text)):
        return 'hu'  # Hungarian
    if bool(re.search(r'[áéíóúà]', text)) and bool(re.search(r'[çã]', text)):
        return 'pt'  # Portuguese
    if bool(re.search(r'[àèéìíîòóùú]', text)):
        return 'it'  # Italian
    if bool(re.search(r'[ěščřžýáíé]', text)):
        return 'cs'  # Czech
    if bool(re.search(r'[äö]', text)) and bool(re.search(r'[aa]', text)):
        return 'fi'  # Finnish
    if bool(re.search(r'[ĳœ]', text)):
        return 'nl'  # Dutch
    
    # Southeast Asian
    if bool(re.search(r'[đáàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữự]', text)):
        return 'vi'  # Vietnamese
    if bool(re.search(r'[ạẹịọụả]', text)):
        return 'id'  # Indonesian (with specific diacritics)
    
    # Other languages with Latin script but distinctive patterns
    if bool(re.search(r'[ğçşıİ]', text)):
        return 'tr'  # Turkish
    
    # Default to English for Latin script without distinctive markers
    return 'en'

@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGE_CODES)

@app.route('/api/translate', methods=['POST'])
def api_translate():
    data = request.json
    text = data.get('text', '')
    source_lang = data.get('source_lang', 'auto')
    target_lang = data.get('target_lang', 'en')
    model_size = data.get('model_size', 'balanced')
    
    # Auto-detect source language if not specified
    if source_lang == 'auto':
        source_lang = detect_language(text)
    
    # Don't translate if languages are the same
    if source_lang == target_lang:
        return jsonify({
            "translated_text": text,
            "detected_lang": source_lang,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "processing_time": 0
        })
    
    result = translate_text(text, source_lang, target_lang, model_size)
    
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict) and 'error' in result[0]:
        return jsonify(result[0]), result[1]
    
    # Add detected language if auto-detection was used
    if data.get('source_lang') == 'auto':
        result['detected_lang'] = source_lang
    
    return jsonify(result)

@app.route('/api/detect', methods=['POST'])
def api_detect():
    data = request.json
    text = data.get('text', '')
    detected_lang = detect_language(text)
    return jsonify({"detected_lang": detected_lang})

@app.route('/api/languages')
def api_languages():
    return jsonify(LANGUAGE_CODES)

@app.route('/api/supported_pairs')
def api_supported_pairs():
    """Return list of supported direct translation pairs"""
    return jsonify({
        "direct_pairs": list(AVAILABLE_DIRECT_PAIRS),
        "fallback": "M2M100 multilingual model supports all language pairs in LANGUAGE_CODES"
    })

if __name__ == '__main__':
    # Try to load the M2M100 model at startup as universal fallback
    try:
        print(f"Running on device: {device}")
        print("Loading M2M100 multilingual model as fallback...")
        m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(device)
        print("M2M100 model loaded successfully")
    except Exception as e:
        print(f"Failed to load M2M100 model: {e}")
    
    # Pre-load common translation models
    print("Pre-loading common translation models...")
    # Focus on higher-resource Indian languages and popular global languages
    for lang in ['hi', 'bn', 'ta', 'fr', 'es', 'de', 'zh', 'ar', 'ru']:
        try:
            load_model('en', lang)
            load_model(lang, 'en')
        except Exception as e:
            print(f"Failed to preload {lang} models: {e}")
    
    # Serve the application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)