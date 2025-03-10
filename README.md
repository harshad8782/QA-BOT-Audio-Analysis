## **📌 Problem Statement**  

### **Description**  
Develop a QA-BOT that **analyzes and evaluates real-time performance** of voice agents by processing their interactions, identifying key metrics (**accuracy, sentiment, responsiveness**), and providing actionable insights to enhance **customer service quality**.  

#### **Input**  
-  **Real-time audio** from customer interactions.  
-  **Call transcripts & metadata** (duration, sentiment, keywords).  
-  **Predefined performance metrics** (e.g., response time, tone, accuracy).  

#### **Output**  
-  **Performance scores & insights** for each agent.  
-  **Alerts for critical issues** (e.g., negative sentiment, long pauses).  

---
### **🔹 Required Solution**  
# **QA-BOT: Real-Time Voice Agent Analysis** 🎙️  

QA-BOT is an AI-powered solution for monitoring and evaluating customer service and BPO agent interactions in **real time**. It processes audio data, analyzes sentiment, detects profanity, evaluates tonality, and provides actionable insights to improve service quality.  

---

## **🚀 Features**  
 **Real-Time Transcription** – Fast & accurate speech-to-text conversion with timestamps.  
 **Speaker Diarization** – Identifies & labels different speakers in a conversation.  
 **Sentiment Analysis** – Detects emotional states & conversation context.  
 **Profanity Detection** – Automatically flags inappropriate language.  
 **Pause Detection** – Analyzes response times & conversation flow.  
 **Tonality Checking** – Evaluates voice tone & speech patterns.  
 **Knowledge Accuracy (Upcoming Feature)** – Assesses information correctness.  

---

## **🛠 Tech Stack**  

| Component             | Technology Used |
|----------------------|----------------|
| **Transcription**    | VOSK ASR + Kaldi Recognizer |
| **Sentiment Analysis** | NLTK's Sentiment Intensity Analyzer (VADER) |
| **Profanity Detection** | Better Profanity Library |
| **Audio Processing** | PyAudio, Pydub |
| **Machine Learning (Future)** | Hugging Face Transformers |

---

## **🔧 Technical Implementation**  

### **1️⃣ Transcription** 📝  
**Key Aspects:** High-speed real-time transcription, accuracy, timestamping.  
- **Approach:**  
  - Captures live audio via **PyAudio**.  
  - Segments audio into frames for processing.  
  - Uses **VOSK + Kaldi Recognizer** for real-time parsing.  
  - Outputs structured transcripts with timestamps.  

### **2️⃣ Speaker Diarization** 🎭  
**Key Aspects:** Multi-speaker detection, agent identification.  
- **Approach:**  
  - Detects speaker change using **pause-based analysis**.  
  - Converts sentences into embeddings for voice pattern comparison.  
  - Identifies agents based on a dictionary of common phrases.  

### **3️⃣ Sentiment Analysis** 😊😡  
**Key Aspects:** Understanding emotional tone, detecting negativity.  
- **Approach:**  
  - Converts speech to text.  
  - Uses **VADER Lexicon (NLTK)** for sentiment polarity.  
  - Labels speech as **Positive, Negative, Neutral, Angry, or Sarcastic**.  

### **4️⃣ Profanity Detection** 🚨  
**Key Aspects:** Detects inappropriate language, raises alerts.  
- **Approach:**  
  - Uses **Better Profanity Library** for real-time monitoring.  
  - Flags offensive words in transcripts.  

### **5️⃣ Tonality Analysis** 🔊  
**Key Aspects:** Evaluates speech tone, conversational style.  
- **Approach:**  
  - Uses **pre-trained NLP models** (e.g., Hugging Face Transformers).  
  - Categorizes tone as **Formal, Casual, Assertive, or Urgent**.  

---
## 📌 Output Examples

### 📝 Real-Time Transcription Output
![Real-Time Transcription](Output/live_inpuput.png)


### 📝 Audio File-Input Transcription Output
![Audio File-Input Transcription 1](Output/file_inpuput.png)


### 📝 Audio File-Input Transcription Output
![Audio File-Input Transcription 1](Output/file_inpuput2.png)

---

## **📥 Setup & Installation**  
### **🔹 Prerequisites**  
Ensure you have **Python 3.8+** installed.  
Install dependencies using:  
```bash
pip install pyaudio vosk nltk better_profanity transformers torch sentencepiece
