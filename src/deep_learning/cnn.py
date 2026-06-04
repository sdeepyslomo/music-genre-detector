import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.a_prcs import get_dataset_path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
                  Conv2D,
                  MaxPooling2D,
                  Flatten,
                  Dense,
                  Dropout,
                  BatchNormalization,
                  GlobalAveragePooling2D
                )
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

#fix randomness
tf.random.set_seed(42)
np.random.seed(42)

#function to convert .wav -> melspec
def extract_mel_spec(file_path):
  audio, sample_rate = librosa.load(file_path)
  #waveform->freq representation
  mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sample_rate,
    n_mels=128,
    hop_length=512
  )
  #db scale
  spec = librosa.power_to_db(
    mel_spec,
    ref=np.max
  )
  return spec,sample_rate

spectrograms=[]
labels=[]

DATASET_PATH = get_dataset_path()
for genre in os.listdir(DATASET_PATH):
  genre_path = os.path.join(DATASET_PATH, genre)
  if not os.path.isdir(genre_path):
    continue
  for file in os.listdir(genre_path):
    file_path = os.path.join(genre_path,file)
    try:
      spec,sample_rate = extract_mel_spec(file_path)

      MAX_LEN = 1500
      if spec.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - spec.shape[1]
        spec = np.pad(
               spec,
               pad_width=((0,0),(0,pad_width)),
               mode='constant'
        )
      else:
        spec = spec[:, :MAX_LEN]

      spectrograms.append(spec)
      labels.append(genre)
    except Exception as e:
      print(f"Error processing {file_path}: {e}")

spectrograms=np.array(spectrograms)
labels=np.array(labels)

#converts label names into integers since CNN undertands integers better than strings
encoder=LabelEncoder()
labels=encoder.fit_transform(labels)

#for adding grayscale
spectrograms=spectrograms[...,np.newaxis]

#normalize spec values
spectrograms = (spectrograms + 80) / 80

#train-test-split
x_train,x_test,y_train,y_test=train_test_split(
  spectrograms,
  labels,
  test_size=0.2,
  random_state=42,
  stratify=labels
)

#The entire CNN model structure
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,1500,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    GlobalAveragePooling2D(),
    Dense(256,activation='relu'),
    Dropout(0.3),
    Dense(10,activation='softmax')
])
#updates parameters after each batch run for better accuracies
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)
#prevents overfitting by stopping training after consecutive bad validation epochs
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
history=model.fit(
  x_train,
  y_train,
  epochs=30,
  batch_size=16,
  validation_data=(x_test,y_test),
  callbacks = [early_stop]
)
spec,sample_rate= extract_mel_spec("data/silvera.wav")
print(spec.shape)

plt.figure(figsize=(10,4))
librosa.display.specshow(
  spec,
  sr=sample_rate,
  x_axis ='time',
  y_axis='mel'
)

history.history['accuracy']
history.history['val_accuracy']
history.history['loss']
history.history['val_loss']

plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.show()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()






