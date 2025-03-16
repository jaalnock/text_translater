from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, DistilBertForSequenceClassification, DistilBertTokenizer
import spacy
import requests
from bs4 import BeautifulSoup
import pytesseract
import cv2
import chardet
import PyPDF2
from pdf2image import convert_from_path
import torch
from PIL import Image as PILImage  # Fixed missing import

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load models
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
context_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
context_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
nlp = spacy.load("en_core_web_lg")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Translation function
def translate_text(text, target_lang):
    tokenizer.src_lang = "en_XX"
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    lang_code = "hi_IN" if target_lang == "hi" else "ta_IN"
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[lang_code])
    translated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    doc = nlp(text)
    key_details = set()
    for ent in doc.ents:
        if ent.label_ in ["GPE", "DATE", "ORG", "PERSON"]:
            if len(ent.text) < 20 and not any(c.isspace() for c in ent.text.strip()) and "GOYA" not in ent.text.upper():
                key_details.add(ent.text)
    return translated, list(key_details)

# Extract text from image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(PILImage.open(image_path))
    return text

# Extract text from file
def extract_text_from_file(file_path, file_name):
    if file_name.lower().endswith('.txt'):
        with open(file_path, "rb") as f:
            result = chardet.detect(f.read())
        encoding = result['encoding'] or 'utf-8'
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    elif file_name.lower().endswith('.pdf'):
        try:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                if text.strip():
                    return text
                else:
                    images = convert_from_path(file_path)
                    return "".join(pytesseract.image_to_string(img) for img in images)
        except Exception as e:
            return f"Error: {str(e)}"
    return "Unsupported file format"

# Scrape text from URL
def scrape_text_from_link(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())[:3500]
        return text if text.strip() else "No meaningful text found"
    except Exception as e:
        return f"Error: {str(e)}"

# Summarize text
def summarize_text(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return " ".join(sentences[:2])

# Enhanced suggest_actions function
def suggest_actions(key_details, original_text, target_lang):
    suggestions_en = []
    
    # Use DistilBERT to classify sentiment/context
    inputs = context_tokenizer(original_text[:512], return_tensors="pt", truncation=True, padding=True)
    outputs = context_model(**inputs)
    sentiment = torch.argmax(outputs.logits, dim=1).item()  # 0 = negative, 1 = positive
    
    # Extract entities
    locations = [d for d in key_details if nlp(d).ents and nlp(d).ents[0].label_ == "GPE"]
    dates = [d for d in key_details if nlp(d).ents and nlp(d).ents[0].label_ == "DATE"]
    orgs = [d for d in key_details if nlp(d).ents and nlp(d).ents[0].label_ == "ORG"]
    people = [d for d in key_details if nlp(d).ents and nlp(d).ents[0].label_ == "PERSON"]
    
    # Analyze text for topics
    doc = nlp(original_text.lower())
    food_related = any(token.text in {"food", "eat", "restaurant", "cuisine", "dining"} for token in doc)
    travel_related = any(token.text in {"visit", "travel", "trip", "explore"} for token in doc)
    event_related = any(token.text in {"event", "schedule", "plan", "meeting"} for token in doc)
    
    # Generate suggestions in English
    if locations:
        if sentiment == 1:  # Positive
            if travel_related:
                suggestions_en.append(f"Plan a trip to {locations[0]} to explore its vibrant culture.")
            if food_related:
                suggestions_en.append(f"Discover local eateries and cuisine in {locations[0]}.")
            suggestions_en.append(f"Visit popular attractions in {locations[0]}.")
            if len(locations) > 1:
                suggestions_en.append(f"Travel from {locations[0]} to {locations[1]} for a unique journey.")
        else:  # Negative
            suggestions_en.append(f"Research travel tips for {locations[0]} to improve your experience.")
    
    if dates:
        if event_related:
            suggestions_en.append(f"Organize an event or outing on {dates[0]}.")
        elif sentiment == 1:
            suggestions_en.append(f"Plan a special activity for {dates[0]}.")
    
    if orgs:
        if food_related:
            suggestions_en.append(f"Check out {orgs[0]} for a dining experience.")
        else:
            suggestions_en.append(f"Explore activities related to {orgs[0]}.")
    
    if people:
        suggestions_en.append(f"Contact {people[0]} for local insights or collaboration.")
    
    # Fallback
    if not suggestions_en and locations:
        suggestions_en.append(f"Look up interesting things to do in {locations[0]}.")
    elif not suggestions_en:
        suggestions_en.append("Consider activities based on your preferences.")
    
    # Limit to 5 unique suggestions in English
    suggestions_en = list(dict.fromkeys(suggestions_en))[:5]
    
    # Translate suggestions into target language
    suggestions_translated = [translate_text(s, target_lang)[0] for s in suggestions_en]
    
    return suggestions_translated

# API Endpoints
@app.route("/process", methods=["POST"])
def process_input():
    input_type = request.form.get("inputType")
    target_lang = request.form.get("language", "hi")

    if input_type == "text":
        text = request.form.get("text")
        if not text:
            return jsonify({"error": "Text is required"}), 400
        translated, key_details = translate_text(text, target_lang)
        suggestions = suggest_actions(key_details, text, target_lang)
        return jsonify({
            "translated": translated,
            "keyDetails": key_details,
            "suggestions": suggestions
        })

    elif input_type in ["file", "image"]:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            text = (extract_text_from_file if input_type == "file" else extract_text_from_image)(file_path, filename)
            if text.startswith("Error"):
                return jsonify({"error": text}), 400
            translated, key_details = translate_text(text, target_lang)
            suggestions = suggest_actions(key_details, text, target_lang)
            os.remove(file_path)
            return jsonify({
                "extractedText": text[:2800],
                "translated": translated,
                "keyDetails": key_details,
                "suggestions": suggestions
            })
        return jsonify({"error": "Invalid file format"}), 400

    elif input_type == "link":
        url = request.form.get("url")
        if not url or not url.startswith("http"):
            return jsonify({"error": "Valid URL required"}), 400
        text = scrape_text_from_link(url)
        if text.startswith("Error"):
            return jsonify({"error": text}), 400
        translated, key_details = translate_text(text, target_lang)
        suggestions = suggest_actions(key_details, text, target_lang)
        return jsonify({
            "extractedText": text[:2800],
            "translated": translated,
            "keyDetails": key_details,
            "suggestions": suggestions
        })

    return jsonify({"error": "Invalid input type"}), 400

@app.route("/summarize", methods=["POST"])
def summarize_input():
    input_type = request.form.get("inputType")
    target_lang = request.form.get("language", "hi")
    text = request.form.get("text") or ""
    
    if input_type == "file" and "file" in request.files:
        file = request.files["file"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        text = extract_text_from_file(file_path, filename)
        os.remove(file_path)
    elif input_type == "image" and "file" in request.files:
        file = request.files["file"]
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        file.save(file_path)
        text = extract_text_from_image(file_path)
        os.remove(file_path)
    elif input_type == "link":
        text = scrape_text_from_link(request.form.get("url", ""))

    if not text:
        return jsonify({"error": "No text to summarize"}), 400
    
    summary_en = summarize_text(text)
    summary_translated, _ = translate_text(summary_en, target_lang)
    return jsonify({
        "summaryEnglish": summary_en,
        "summaryTranslated": summary_translated
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)