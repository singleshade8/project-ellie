import openai
from dotenv import load_dotenv
import os
from transformers import pipeline
import joblib
import torch

# Load .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load ASR
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# Load sentiment
sentiment = pipeline("sentiment-analysis")

# Load intent
label_encoder = joblib.load("label_encoder.pkl")
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
intent_model = DistilBertForSequenceClassification.from_pretrained("intent_model").to("cuda" if torch.cuda.is_available() else "cpu")

# Load sample
result = asr("sample_call.wav")
text = result.get("text", "").strip()
print(f"📝 Transcribed: {text}")

# Sentiment
s = sentiment(text)[0]
print(f"😊 Sentiment: {s['label']}")

# Intent
inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    logits = intent_model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    intent = label_encoder.inverse_transform([pred])[0]
print(f"🎯 Intent: {intent}")

# MOCKED GPT Suggestion

def get_suggestion(text, sentiment, intent):
    return f"(STATIC) Based on intent '{intent}' and sentiment '{sentiment}', recommend a polite response."

suggestion = get_suggestion(text, s['label'], intent)
print(f"💡 GPT Suggestion: {suggestion}")
