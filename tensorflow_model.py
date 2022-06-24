import speech_recognition
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from keras.models import Sequential
from preprocess import *

testdb = pd.read_csv("result.csv")

input_data = np.array(testdb["text"])
output_data = np.array(testdb["answer"])
cv = CountVectorizer()
input_data = cv.fit_transform(input_data)

xtrain, xtest, ytrain, ytest = train_test_split(input_data, output_data, test_size=0.3, random_state=5)

model = Sequential()
model.add(Dense(units=2, activation = 'relu'))
model.add(Dense(units=1, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(xtrain, ytrain, epochs=5, validation_data=(xtest, ytest))

sr = speech_recognition.Recognizer()
sr.pause_threshold = 0.5

with speech_recognition.Microphone() as mic:
    sr.adjust_for_ambient_noise(source=mic, duration=0.5)
    audio = sr.listen(source=mic)
    query = sr.recognize_google(audio_data=audio, language='ru-RU').lower()

voice = query
testdb = cv.transform([voice]).toarray()
print(voice)
print(model.predict(testdb))