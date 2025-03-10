import os
import json
import numpy as np
import nltk
import wave
from vosk import Model, KaldiRecognizer
from nltk.sentiment import SentimentIntensityAnalyzer
from better_profanity import profanity
from pydub import AudioSegment, silence

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

# 🔹 Speaker tracking variables
energy_thresholds = []
energy_window_size = 5
last_speaker = None
speaker_count = 1

# 🔹 Function to check profanity
def check_profanity(text):
    """Detect but do NOT censor profanity. Returns a flag."""
    return profanity.contains_profanity(text)

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

# 🔹 Function to process audio file
def process_audio_file(file_path):
    """Processes an audio file for transcription, speaker tracking, sentiment, and tonality."""
    print("\n🔍 Processing audio file...")

    # Convert audio to WAV (if needed)
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Break audio into speech segments
    chunks = silence.split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
    
    results = []
    full_transcript = []
    start_time = 0
    cuss_words_detected = []

    for chunk in chunks:
        raw_audio = chunk.raw_data
        if rec.AcceptWaveform(raw_audio):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            words = result.get("result", [])

            if text:
                sentiment = analyze_sentiment(text)
                tonality = analyze_tonality(text)
                speaker = identify_speaker(raw_audio)

                # Store transcript
                full_transcript.append(f"[{round(start_time, 2)}s] {speaker}: {text}")

                # Check for profanity
                if check_profanity(text):
                    cuss_words_detected.append(f"[{round(start_time, 2)}s] 🚨 Profanity detected!")

                # Store sentiment & tonality flags
                results.append({
                    "start_time": round(start_time, 2),
                    "speaker": speaker,
                    "sentiment": sentiment,
                    "tonality": tonality,
                })
        
        # Update time index
        start_time += len(chunk) / 1000  

    return results, full_transcript, cuss_words_detected

# 🔹 Get user input for audio file
if __name__ == "__main__":
    audio_file = input("🎵 Enter path to audio file: ").strip()

    if not os.path.exists(audio_file):
        print("❌ Error: File not found!")
        exit(1)

    # Process the audio file
    results, full_transcript, cuss_words_detected = process_audio_file(audio_file)
    
    print("\n✅ Processing Complete!")
    
    # 📜 Display Full Transcript
    print("\n📜 Full Transcript:\n")
    for line in full_transcript:
        print(line)

    # 🚨 Display Detected Flags
    if cuss_words_detected or results:
        print("\n🚨 Detected Flags:\n")

        for entry in results:
            print(f"🔊 **Start Time:** {entry['start_time']}s")
            print(f"   🗣 **Speaker:** {entry['speaker']}")
            print(f"   🏷 **Sentiment:** {entry['sentiment']}")
            print(f"   🎭 **Tonality:** {entry['tonality']}")
            print("-" * 40)

        for flag in cuss_words_detected:
            print(flag)

    else:
        print("✅ No flags detected in the audio.")
