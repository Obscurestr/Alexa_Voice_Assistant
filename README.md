#  AI Voice Assistant

An AI-powered voice assistant built with Python that records user speech, converts it to text, predicts the user's intent using a machine learning model, performs the corresponding action, and responds with synthesized speech.

##  Features

-  Record voice commands using a microphone
-  Speech-to-text transcription with OpenAI Whisper
-  Intent classification using Machine Learning
-  Text-to-speech responses
-  Open websites
-  Play music
-  Tell jokes

---

##  Technologies Used

- Python
- OpenAI Whisper
- Scikit-learn
- TF-IDF Vectorizer
- MLPClassifier
- Pandas
- NumPy
- Pyttsx3
- SoundDevice
- SciPy

---

##  Project Structure

```
AI_Voice_Assistant/
│
├── Alexa_Voice_Assistant.py   # Main assistant logic
├── alexa_data.csv             # Dataset for intent classification
├── input.wav                  # Recorded audio
├── requirements.txt
└── README.md
```

---

##  Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/AI-Voice-Assistant.git
cd AI-Voice-Assistant
```

### Create a virtual environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

##  Machine Learning Pipeline

The assistant uses Natural Language Processing (NLP) to classify spoken commands.

### Workflow

1. Record audio from microphone
2. Convert speech to text using Whisper
3. Transform text into numerical features using TF-IDF
4. Predict the user's intent using an MLPClassifier
5. Execute the corresponding action
6. Respond using text-to-speech

---

##  Model Evaluation

The intent classifier was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

##  Dataset

The dataset (`alexa_data.csv`) contains:

- User commands
- Corresponding intent labels

Example:

| Command | Intent |
|---------|--------|
| Play some music | play_music |
| Open YouTube | open_website |
| Tell me a joke | jokes_fun |

---

##  Example Commands

- "Play some music"
- "Open YouTube"
- "Tell me a joke"
- "Open Google"
- "Play jazz music"

---

##  Future Improvements

- Add weather information
- Voice authentication
- Integration with ChatGPT API
- Smart home control
- Reminder and calendar support
- Better intent classification using transformer models
- Support for multiple languages

---

##  Author

**Sara Trnjakov**

Graduate in Applied Information Technologies

Interested in:

- Data Science
- Machine Learning
- Artificial Intelligence
- NLP

---

##  License

This project is intended for educational and portfolio purposes.
