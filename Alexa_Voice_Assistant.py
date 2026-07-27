#!/usr/bin/env python
# coding: utf-8

# # Alexa Voice Assistant

# ## 1. Importing Libraries and Installing Dependencies

# This section imports all the required libraries for the voice assistant, including audio recording, speech recognition, text processing, machine learning, and text-to-speech functionality. It also displays the active Python environment (sys.prefix) and installs all necessary dependencies using pip to ensure the application runs correctly.

# In[98]:


import sys
import sounddevice as sd
from scipy.io.wavfile import write
import os
import subprocess
import urllib
import whisper
import soundfile as sf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import webbrowser
import datetime
import random
import re
import pyttsx3
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score
import seaborn as sns
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import json


# In[2]:


print(sys.prefix)


# In[1]:


 


# ## 2. Audio Recording

# This function records the user's voice through the microphone for a specified duration and saves it as a WAV audio file. It uses the selected input and output audio devices, waits until the recording is complete, and stores the recording for further speech recognition processing.

# In[4]:


def record_audio(filename="input.wav",duration=5,fs=16000):
    print("🎤 Listening...")
    recording=sd.rec(int(duration*fs),samplerate=fs,channels=1, dtype='int16')
    sd.wait()
    write(filename,fs,recording)
    print("✅ Audio recorded")


# In[5]:


sd.query_devices()


# In[6]:


sd.default_device=(14,12)


# In[7]:


record_audio()


# ## 3. FFmpeg Setup

# This section downloads FFmpeg, creates the required directory if it does not already exist, and adds the FFmpeg executable to the system PATH. This ensures that the Whisper speech recognition model can process audio files correctly.

# In[8]:


url="https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


# In[9]:


os.makedirs(r"C:\Users\sarat\Downloads\alexa_project/ffmpeg",exist_ok=True)


# In[10]:


urllib.request.urlretrieve(url,r"C:\Users\sarat\Downloads\alexa_project/ffmpeg/ffmpeg.zip")


# In[11]:


os.environ["PATH"]


# In[12]:


ffmpeg_path=r"C:\Users\sarat\Downloads\alexa_project\ffmpeg\ffmpeg\ffmpeg-8.1.2-essentials_build\bin"


# In[13]:


os.environ["PATH"]=os.environ["PATH"]+os.pathsep+ffmpeg_path


# ## 4. Speech-to-Text Conversion

# This section installs the required audio library, loads the Whisper base model, and defines a function that converts recorded speech into text. The audio file is loaded, processed, and transcribed, with the recognized text printed and returned for further processing.

# In[14]:


 


# In[180]:


whisper_model = whisper.load_model("base")


# In[181]:


def speech_to_text(audio_file):
    audio, fs = sf.read(
    r"C:\Users\sarat\Downloads\alexa_project\input.wav"
    )
    audio = audio.astype(np.float32)
    result = whisper_model.transcribe(audio,fp16=False)
    print(result['text'])
    return result['text']


# ## 5. Intent Classification Model

# This section loads the Alexa command dataset, transforms text prompts into numerical features using TF-IDF vectorization, and trains an MLP neural network classifier to recognize user intents. The trained model predicts the intent of a new voice command, allowing the assistant to determine the appropriate action to perform.

# In[141]:


alexa_df=pd.read_csv(r"C:\Users\sarat\Downloads\alexa_project\alexa_data.csv")


# In[142]:


alexa_df.head()


# In[143]:


alexa_df.shape


# In[144]:


vectorizer=TfidfVectorizer()


# In[145]:


X=vectorizer.fit_transform(alexa_df["prompt"])


# In[146]:


y=alexa_df["intent"]


# In[147]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[148]:


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Linear SVM": LinearSVC(random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(50,25),
                         max_iter=500,
                         random_state=42)
}


# In[149]:


results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "F1-score": f1
    })


# In[150]:


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)
results_df["Accuracy"] = results_df["Accuracy"].round(3)
results_df["F1-score"] = results_df["F1-score"].round(3)
results_df


# All evaluated machine learning models achieved 100% accuracy and F1-score on the test dataset. This indicates that the intent classes are highly separable and that multiple classification algorithms can successfully recognize user commands. Since all models performed equally well, the MLP classifier was selected for the final implementation of the voice assistant.

# In[151]:


intent_model=MLPClassifier(
    hidden_layer_sizes=(50,25),
    max_iter=500,
    random_state=42
)


# In[152]:


intent_model.fit(X_train, y_train)


# In[153]:


def predict_intent(text):
    X_new=vectorizer.transform([text])
    intent=intent_model.predict(X_new)[0]
    print("Detected Intent:",intent)
    return intent


# In[154]:


predict_intent("Set Alarm For 7 PM")


# In[155]:


predict_intent("Play Song For Me")


# In[156]:


predict_intent("Add milk to my shopping list")


# In[157]:


predict_intent("What is the news today?")


# ## 6. Model Evaluation 

# The trained intent classification model is evaluated on the test dataset using precision, recall, F1-score, and accuracy metrics. A confusion matrix is also generated to visualize the model's predictions and identify potential classification errors between different user intents.

# In[158]:


y_pred = intent_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[159]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Intent Classification Confusion Matrix")
plt.show()


# The MLP classifier achieved 100% accuracy on the held-out test set, showing that the intents in the dataset are highly separable. Additional testing with custom user queries confirmed that the model can correctly classify unseen commands.

# ## 7. Intent-Based Action Execution

# This section defines the assistant's response logic by mapping detected user intents to specific actions, such as opening websites, playing music, providing information, setting reminders, or performing calculations. The function processes the predicted intent and user input, executes the corresponding task, and returns an appropriate response.

# In[160]:


MEMORY_FILE = r"C:\Users\sarat\Downloads\alexa_project\memory.json"


# In[161]:


def load_memory():
    with open("memory.json", "r") as file:
        return json.load(file)


# In[162]:


memory = load_memory()
def perform_action(intent,prompt=""):
    if intent=="play_music":
        webbrowser.open("https://www.youtube.com/results?search_query=music")
        return "Playing music on youtube"
    elif intent=="open_website":
        prompt_lower=prompt.lower()
        cleaned=re.sub(r'[a-zA-Z0-9\s]','',prompt_lower)
        if "youtube" in cleaned or "you tube":
            webbrowser.open("https://www.youtube.com/")
            return "Opening Youtube"
        elif "google" in prompt_lower:
            webbrowser.open("https://www.google.com/")
            return "Opening Google"
        elif "github" in prompt_lower:
            webbrowser.open("https://github.com/")
            return "Opening Github"
        else:
            return "Which website would you like me to open?"
    elif intent=="jokes_fun":
        jokes=[
            "Why did the computer get cold? Because it forgot to close the Window!",
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "I told my AI assistant a joke. It said, processing humor module."
        ]
        return random.choice(jokes)
    elif intent=="news":
        webbrowser.open("https://news.google.com/")
        return "Here's today top headlines."
    elif intent=="movies":
        webbrowser.open("https://www.imdb.com/chart/top/")
        return "Here's todays top movies."
    elif intent=="set_timer":
        minutes=re.findall(r'\d+',prompt)
        minutes=int(minutes[0]) if minutes else 1
        return f"Timer set for {minutes} minutes (simulation)."
    elif intent=="alarm":
        time_match=re.search(r'(\d{1,2}(?::(\d{2}))?\s*(am|pm)?',prompt.lower())
        if time_match:
            hour=int(time_match.group(1))
            minute=int(time_match.group(2) if time_match.group(2) else 0)
            period=time_match.group(3) if time_match.group(3) else "am"
            return f"Alarm set for {hour}:{str(minute).zfill(2)} {period.upper()} simulation"
        return "Alarm set.(simulation)"
    elif intent=="reminder":
        return "Reminder saved.(simulation)"
    elif intent=="date_time":
        now=datetime.datetime.now()
        return f"It's {now.strftime('%I:%M: %p')} on {now.strftime('%B %d, %Y')}"
    elif intent=="calendar":
        return "You have no events today.(simulation)"
    elif intent=="weather":
        prompt_lower=prompt.lower()
        match=re.search(r"(in|of)\s+([a-zA-Z\s]+)",prompt_lower)
        if match:
            city=match.group(2).strip()
            webbrowser.open(f"https://www.google.com/search?q=weather+in+{city.replace(' ','+')}")
            return f"Showing current weather in {city.title()}"
        else:
            webbrowser.open(f"https://www.google.com/search?q=weather+today")
            return "Here's the latest weather forecast."
    elif intent=="general_qa":
        prompt_lower=prompt.lower()
        if any(word in prompt_lower for word in ["temperature","weather","forecast"]):
            match=re.search(r"(in|of)\s+([a-zA-Z\s]+)",prompt_lower)
            if match:
                city=match.group(2).strip()
                webbrowser.open(f"https://www.google.com/search?q=weather+in+{city.replace(' ','+')}")
                return f"Showing current weather in {city.title()}"
        query=prompt.replace(" ","+")
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return "Here's what I found  on the web."
    elif intent=="facts":
        facts=[
            "Honey never spoils.",
            "Octopuses have three hearts.",
            "Tomato is a fruit."
        ]
        return random.choice(facts)
    elif intent=="traffic":
        webbrowser.open(f"https://www.google.com/maps/dir/Home/Office")
        return "Checking traffic on a usual rute."
    elif intent=="directions":
        webbrowser.open(f"https://www.google.com/maps/search/{location}")
        return f"Showing directions to {location}."
    elif intent=="shopping_list":
        item=prompt.lower().replace("to my shopping list","").strip()
        memory["shopping_list"].append(item)
        save_memory(memory)
        return f"Added {item} to your shopping list."
    elif intent=="smart_home":
        return "Command executed.(simulation)"
    elif intent=="personality":
        responses=[
            "I'm doing great!Ready to help you.",
            "I'm your AI assistant!",
            "I waqs created to make your life easier."
        ]
        return random.choice(responses)
    elif intent=="calculator":
        nums=list(map(int,re.findall(r'\d+',prompt)))
        if "plus" in prompt and len(nums)==2:
            return f"The answer is {nums[0]+nums[1]}"
        elif "minus" in prompt and len(nums)==2:
            return f"The answer is {nums[0]-nums[1]}"
        elif "times" in prompt and len(nums)==2:
            return f"The answer is {nums[0]*nums[1]}"
        elif "divide" in prompt and len(nums)==2:
            return f"The answer is {nums[0]/nums[1]}"
    elif intent=="memory":
        memory = load_memory()
        response = f"Your name is {memory['name']}.\n"
        response += f"Your favorite music is {memory['favorite_music']}.\n"
        response += "Your shopping list contains:\n"
        for item in memory["shopping_list"]:
            response += f"- {item}\n"
        return response
    else:
        return "Sorry,I didn't understand the command."



# In[163]:


user_text="Play song for me."


# In[164]:


intent=predict_intent(user_text)
intent


# In[165]:


perform_action(intent,user_text)


# In[36]:


user_text2="Who is the prime minister of Serbia"


# In[37]:


intent2=predict_intent(user_text2)
intent2


# In[38]:


perform_action(intent2,user_text2)


# In[166]:


user_text = "What do you remember about me?"


# In[167]:


intent = predict_intent(user_text)


# In[168]:


response = perform_action(intent, user_text)
print(response)


# ## 8. Persistent Memory

# The assistant stores user preferences, including the user's name, favorite music genre, and shopping list, in a JSON file. This allows information to persist across multiple sessions, providing a more personalized user experience.

# In[169]:


def save_memory(memory):
    with open("memory.json", "w") as file:
        json.dump(memory, file, indent=4)


# In[170]:


memory = load_memory()


# In[171]:


def remember_name(name):
    memory["name"] = name
    save_memory(memory)


# In[172]:


remember_name("Sara")


# In[173]:


memory["favorite_music"]="Metal"
save_memory(memory)


# ## 9. Voice Output and Assistant Pipeline

# This section initializes the text-to-speech engine using pyttsx3 and configures the voice, speech rate, and volume settings. The run_assistant() function connects all components of the system by recording audio, converting speech to text, predicting the user's intent, performing the corresponding action, and generating a spoken response.

# In[174]:


engine=pyttsx3.init("sapi5")


# In[175]:


voices=engine.getProperty("voices")


# In[176]:


engine.setProperty("voice",voices[1].id)
engine.setProperty("rate",165)
engine.setProperty("volumn",1.0)


# In[177]:


def speak(text):
    print("Alexa:",text)
    engine.say(text)
    engine.runAndWait()


# In[178]:


def run_assistant():
    if memory["name"] != "":
        speak(f"Welcome back {memory['name']}")
    record_audio("input.wav")
    text=speech_to_text("input.wav")
    intent=predict_intent(text)
    response=perform_action(intent,text)
    speak(response)




# In[183]:


run_assistant()

