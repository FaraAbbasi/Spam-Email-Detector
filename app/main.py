import pickle
import os
import re
import string 
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

# Load model + tokenizer once when server starts
model_path = os.path.join("models", "best_spam_model.keras")
tokenizer_path = os.path.join("models", "tokenizer.pickle")


# Load model and tokenizer
model = tf.keras.models.load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

print(f"Model loaded successfully from {model_path}")
print(f"Tokenizer loaded successfully from {tokenizer_path}")

# Helper function to clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"\d+", " num ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Request/Response Scheme
class EmailInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Subject: Following up on our previous conversation...\nHi there, I noticed some SEO issues..."
            }
        }

# Initialize FastAPI app
app = FastAPI(
    title="Email Spam Detection API",
    description="A GRU-powered deep learning API to detect spam emails.",
    version="1.0"
)

# Allow the frontend (served from anywhere, e.g. a local file or dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_index():
    # Read and return the HTML file content
    path = os.path.join("templates", "index.html")
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


@app.get("/health")
def health_check():
    return {"status": "Ok", "message": "Spam/Ham classifier API is running"}

@app.post("/predict")
def predict(email: EmailInput):
    # Clean and preprocess the text
    cleaned_text = clean_text(email.text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, 
        maxlen=150, 
        padding='post', 
        truncating='post'
    )

    # Model prediction
    probs = float(model.predict(padded_sequence, verbose=0).flatten()[0])
    is_spam = probs >= 0.5  # Moved this BEFORE using it
    label = "SPAM 🚨" if is_spam else "HAM ✅"

    return {
        "Email": email.text[:80] + "..." if len(email.text) > 80 else email.text,
        "Label": label,
        "Confidence": round(probs if is_spam else 1.0 - probs, 4),
        "SpamProbability": round(probs, 4)
    }
