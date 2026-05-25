import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils.a_prcs import get_dataset_path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#function to convert .wav -> melspec
def extract_mel_spec(file_path):
  audio, sample_rate = librosa.load(file_path)
  #waveform->freq representation
  mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sample_rate
  )
  #db scale
  mel_spec_db = librosa.power_to_db(
    mel_spec,
    ref=np.max
  )
  return mel_spec_db,sample_rate

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
      spec=spec[:,:1500]
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

#train-test-split
x_train,x_test,t_train,y_test=train_test_split(
  spectrograms,
  labels,
  test_size=0.2,
  random_state=42
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
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.show()



