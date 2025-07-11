import os
import torch
import sounddevice as sd
import numpy as np
import webrtcvad
import queue
import sys
import wave
from transformers import pipeline
import joblib
import requests
from dotenv import load_dotenv

# Load .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Constants
RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
MIN_CHUNK_DURATION = 0.5  # seconds
MAX_CHUNK_DURATION = 10.0  # seconds
VAD_MODE = 3  # strict filtering

# Debug device info
print("Torch device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# VAD
vad = webrtcvad.Vad(VAD_MODE)

# ASR pipeline
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16  # use float16 for GPU
)

# Sentiment analysis
sentiment = pipeline(
    "sentiment-analysis",
    device=0 if torch.cuda.is_available() else -1
)

# Intent classification
try:
    label_encoder = joblib.load("label_encoder.pkl")
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    intent_model = DistilBertForSequenceClassification.from_pretrained("intent_model").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("✅ Intent model loaded")
except:
    label_encoder = None
    intent_model = None
    print("⚠️ Intent model missing, continuing without")

q = queue.Queue()

# Gemini Suggestion
def get_suggestion_from_gemini(intent, sentiment, emotion, text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"A customer just said: \"{text}\"\nSentiment: {sentiment}\nIntent: {intent}\nEmotion: {emotion}\n\nGive a short, polite, actionable suggestion for a customer service agent to respond."
                    }
                ]
            }
        ]
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# Gemini Summary
def summarize_call(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Summarize this customer support conversation in 3–5 bullet points:\n{text}"
                    }
                ]
            }
        ]
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"⚠️ Failed to summarize: {e}"

# Audio callback
def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(indata.copy())

# Main execution
print("\nAvailable input devices:")
for idx, dev in enumerate(sd.query_devices()):
    if dev["max_input_channels"] > 0:
        print(f"[{idx}] {dev['name']}")
mic_index = int(input("👉 Enter mic index to use: "))
print(f"🎙️ Listening on {'cuda' if torch.cuda.is_available() else 'cpu'}...")

transcript_log = []

with sd.InputStream(callback=callback, channels=1, samplerate=RATE, blocksize=FRAME_SIZE, device=mic_index):
    triggered = False
    frames = []
    try:
        while True:
            frame = q.get()
            pcm_data = (frame * 32768).astype(np.int16).tobytes()
            is_speech = vad.is_speech(pcm_data, RATE)

            if is_speech:
                if not triggered:
                    print("🎙️ Voice detected, recording...")
                    triggered = True
                    frames = []
                frames.append(pcm_data)

                if len(b''.join(frames)) / 2 / RATE > MAX_CHUNK_DURATION:
                    print("🛑 Max chunk duration reached")
                    triggered = False

            elif triggered:
                triggered = False
                chunk = b''.join(frames)
                duration = len(chunk) / 2 / RATE
                float_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(float_data**2))
                db = 20 * np.log10(rms + 1e-8)
                print(f"🛑 Voice stopped, chunk duration: {duration:.2f}s, level: {db:.1f} dB")

                if duration < MIN_CHUNK_DURATION:
                    print("⚠️ Chunk too short, skipping")
                    continue
                if db < -45:
                    print("⚠️ Very low voice level, skipping")
                    continue

                fname = "debug_chunk.wav"
                with wave.open(fname, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(RATE)
                    wf.writeframes(chunk)
                print(f"✅ Saved {fname}")

                try:
                    result = asr(fname, generate_kwargs={"language": "en"})
                    text = result.get("text", "").strip()
                    print(f"✅ Final text: {text}")
                    if len(text.split()) < 3:
                        print("⚠️ Too short to analyze, skipping")
                        continue

                    transcript_log.append(text)

                    s = sentiment(text)[0]
                    print(f"Sentiment: {s}")

                    if intent_model:
                        inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
                        with torch.no_grad():
                            logits = intent_model(**inputs).logits
                            probs = torch.softmax(logits, dim=1)
                            pred = torch.argmax(probs, dim=1).item()
                            label = label_encoder.inverse_transform([pred])[0]
                            print(f"Intent: {label}")
                    else:
                        label = "unknown"

                    # Emotion placeholder (removed real detection)
                    emotion = "n/a"

                    suggestion = get_suggestion_from_gemini(label, s['label'], emotion, text)
                    print(f"💡 Suggestion: {suggestion}")

                except Exception as e:
                    print(f"⚠️ Transcription failed: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Call ended, generating summary...")
        full_conversation = " ".join(transcript_log)
        with open("full_transcript.txt", "w", encoding="utf-8") as f:
            f.write(full_conversation)
        summary = summarize_call(full_conversation)
        print("\n📋 Summary:")
        print(summary)
        with open("call_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        sys.exit(0)
