import os
import json
import time
import pyaudio
import numpy as np
import nltk
from vosk import Model, KaldiRecognizer
from nltk.sentiment import SentimentIntensityAnalyzer
from better_profanity import profanity

# 🔹 Download required NLTK data (only once)
nltk.download("vader_lexicon")

# 🔹 Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# 📌 Load Vosk ASR model
model_path = "vosk-model-small-en-us-0.15"
if not os.path.exists(model_path):
    print(f"❌ Error: Vosk model not found at '{model_path}'.")
    exit(1)

model = Model(model_path)
rec = KaldiRecognizer(model, 16000)
rec.SetWords(True)

# 📌 Load profanity filter
profanity.load_censor_words()

# 🔹 Initialize PyAudio for live recording
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)

# 🔹 Speaker tracking variables
energy_thresholds = []
energy_window_size = 5
last_speaker = None
speaker_count = 1

print("\n🎤 Real-time transcription with sentiment & tonality analysis...\n")

# 🔹 Function to check profanity
def check_profanity(text):
    """Detect but do NOT censor profanity. Returns original text and a flag."""
    contains_profanity = profanity.contains_profanity(text)
    return text, contains_profanity  # Return original text & profanity flag

# 🔹 Function to analyze sentiment
def analyze_sentiment(text):
    """Analyze sentiment and detect emotional tone."""
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if scores["pos"] > 0.3 and compound < 0:
        return "Sarcasm"
    elif compound >= 0.05:
        return "Positive"
    elif compound <= -0.6:
        return "Anger"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# 🔹 Function to analyze tonality
def analyze_tonality(text):
    """Determine the tonality of the speech."""
    text_lower = text.lower()

    if any(word in text_lower for word in ["please", "kindly", "would you mind", "if possible"]):
        return "Formal"
    elif any(word in text_lower for word in ["hey", "yo", "what's up", "nah"]):
        return "Casual"
    elif any(word in text_lower for word in ["now", "urgent", "immediately", "asap"]):
        return "Urgent"
    elif any(word in text_lower for word in ["must", "have to", "need to", "should"]):
        return "Assertive"
    
    return "Neutral"

# 🔹 Function to identify speakers based on audio energy
def identify_speaker(audio_chunk):
    """Improved speaker detection using rolling energy averages."""
    global last_speaker, speaker_count, energy_thresholds

    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    energy = np.sum(np.abs(audio_data))

    energy_thresholds.append(energy)
    if len(energy_thresholds) > energy_window_size:
        energy_thresholds.pop(0)

    avg_energy = np.mean(energy_thresholds)

    if last_speaker is None or energy > avg_energy * 1.5:
        last_speaker = f"Speaker {speaker_count}"
        speaker_count = 1 if speaker_count == 2 else 2
    return last_speaker

# 🔹 Real-time transcription loop
try:
    while True:
        try:
            data = stream.read(4000, exception_on_overflow=False)
        except IOError as e:
            print("⚠️ Audio Overflow Error:", e)
            continue  # Skip this iteration

        speaker = identify_speaker(data)

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                original_text, has_profanity = check_profanity(text)
                sentiment = analyze_sentiment(original_text)
                tonality = analyze_tonality(original_text)

                # Add 🚨 flag if profanity is detected
                profanity_flag = "🚨 [Profanity Detected]" if has_profanity else ""

                print(f"{speaker}: {original_text} {profanity_flag} | 🏷 Sentiment: {sentiment} | 🎭 Tonality: {tonality}")
        else:
            partial = json.loads(rec.PartialResult())
            text = partial.get("partial", "")
            if len(text.split()) > 3:
                print(f"{speaker} (Partial): {text}", end="\r")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Transcription stopped.")

# 📌 Cleanup resources
stream.stop_stream()
stream.close()
p.terminate()
