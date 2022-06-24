import speech_recognition
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

testdb = pd.read_csv("result.csv")

v = TfidfVectorizer()
input_data = np.array(testdb["text"])
output_data = np.array(testdb["answer"])
input_data = v.fit_transform(input_data)

xtrain, xtest, ytrain, ytest = train_test_split(input_data, output_data, test_size=0.2, random_state=42, stratify=output_data)
model = MultinomialNB()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
print("Говорите")


sr = speech_recognition.Recognizer()
sr.pause_threshold = 0.5

with speech_recognition.Microphone() as mic:
    sr.adjust_for_ambient_noise(source=mic, duration=0.5)
    audio = sr.listen(source=mic)
    query = sr.recognize_google(audio_data=audio, language='ru-RU').lower()

voice = query
testdb = v.transform([voice]).toarray()
print("Вы сказали:", voice)
print(model.predict(testdb))