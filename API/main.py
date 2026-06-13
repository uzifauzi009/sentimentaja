from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os

# Setup model & preprocessing (dijalankan sekali saat startup)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/indobert")

# Load tokenizer & model 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # mode inference

# Preprocessing tools 
nltk.download('stopwords', quiet=True)
stopwords_id = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_id]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# FastAPI app
app = FastAPI(
    title="SentimenAja API",
    description="API Analisis Sentimen Ulasan Berbahasa Indonesia menggunakan IndoBERT",
    version="1.0.0"
)

class ReviewInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    sentiment: str
    confidence: float

@app.post("/predict", response_model=SentimentOutput)
def predict(review: ReviewInput):
    # Preprocessing
    cleaned = preprocess_text(review.text)

    # Tokenisasi
    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    # Inferensi (tanpa gradient)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        confidence, pred_class = torch.max(probabilities, dim=-1)

    # Konversi label
    label_map = {0: "negatif", 1: "positif"}
    sentiment = label_map[pred_class.item()]
    confidence_score = confidence.item()
    
    return SentimentOutput(sentiment=sentiment, confidence=confidence_score)

@app.get("/")
def root():
    return {"message": "Selamat datang di SentimenAja API! Gunakan endpoint /predict untuk analisis sentimen."}