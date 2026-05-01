import os
import librosa    
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

DATASET_PATH = "data"

features = []
labels = []

for genre in os.listdir(DATASET_PATH):
    genre_path = os.path.join(DATASET_PATH, genre)

    if not os.path.isdir(genre_path):
        continue 
    
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)

        try:
            audio, sample_rate = librosa.load(file_path)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            features.append(mfccs_mean)
            labels.append(genre)

            print(f"Processed: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

print("Total samples:", len(features))     


x = np.array(features)
y = np.array(labels)

print("x shape:", x.shape)
print("y shape:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2, random_state=42
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_predict))

file_path = "data/silvera.wav"
audio, sample_rate = librosa.load(file_path)
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
mfccs_mean = np.mean(mfccs,axis=1)
mfccs_mean = mfccs_mean.reshape(1,-1)
prediction = model.predict(mfccs_mean)
print("Predicted genre:", prediction[0])




    
